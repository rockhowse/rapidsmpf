/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Read a single chunk from a parquet source and send it to an output channel.
 *
 * @param ctx The execution context to use.
 * @param ch_out Channel to which `TableChunk`s are sent.
 * @param stream The stream on which to read the chunk.
 * @param source The `cudf::io::source_info` describing the data to read.
 * @param columns Named columns to read from the file.
 * @param skip_rows Number of rows to skip from the beginning of the file.
 * @param num_rows Number of rows to read.
 * @param predicate Optional predicate to apply during the read.
 * @param sequence_number The ordered chunk id to reconstruct original ordering of the
 * data.
 *
 * @note The caller is responsible for scheduling this coroutine onto a thread pool for
 * execution.
 *
 * @return Streaming node representing the asynchronous read of a chunk and send to the
 * output channel.
 */
Node read_parquet_chunk(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<ThrottlingAdaptor> ch_out,
    rmm::cuda_stream_view stream,
    cudf::io::parquet_reader_options options,
    std::uint64_t sequence_number
) {
    auto ticket = co_await ch_out->acquire();
    auto result = std::make_unique<TableChunk>(
        sequence_number,
        cudf::io::read_parquet(options, stream, ctx->br()->device_mr()).tbl,
        stream
    );
    auto [_, receipt] = co_await ticket.send(std::move(result));
    // Move this coroutine to the back of the queue so that when we release the semaphore
    // it is likely to occur on a different thread, releasing the semaphore resumes any
    // waiters on the current thread which is not what we typically want for throttled
    // reads, we want the next waiting read task to run on a different thread.
    co_await ctx->executor()->yield();
    co_await receipt;
}

}  // namespace

Node read_parquet(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::ptrdiff_t max_tickets,
    cudf::io::parquet_reader_options options,
    cudf::size_type num_rows_per_chunk
) {
    ShutdownAtExit c{ch_out};
    auto throttle = std::make_shared<ThrottlingAdaptor>(ch_out, max_tickets);
    co_await ctx->executor()->schedule();
    auto size = static_cast<std::size_t>(ctx->comm()->nranks());
    auto rank = static_cast<std::size_t>(ctx->comm()->rank());
    auto source = options.get_source();
    RAPIDSMPF_EXPECTS(
        source.type() == cudf::io::io_type::FILEPATH, "Only implemented for file sources"
    );
    // TODO: To handle this we need the options object to allow int64 num_rows and a
    // prefix scan across all the ranks of the total number of rows that would be read by
    // previous ranks.
    RAPIDSMPF_EXPECTS(
        size == 1 || !options.get_num_rows().has_value(),
        "Reading subset of rows not yet supported in multi-rank execution"
    );
    // TODO: To handle this we need a prefix scan across all the ranks of the total number
    // of rows that would be read by previous ranks.
    RAPIDSMPF_EXPECTS(
        size == 1 || options.get_skip_rows() == 0,
        "Skipping rows not yet supported in multi-rank execution"
    );
    auto files = source.filepaths();
    RAPIDSMPF_EXPECTS(
        files.size() < std::numeric_limits<int>::max(), "Trying to read too many files"
    );
    int files_per_rank =
        static_cast<int>(files.size() / size + (rank < (files.size() % size)));
    int file_offset = 0;
    for (auto i = std::size_t{0}; i < rank; i++) {
        file_offset +=
            static_cast<int>(files.size() / size + (i < (files.size() % size)));
    }
    files = std::vector(
        files.begin() + file_offset, files.begin() + file_offset + files_per_rank
    );
    int files_per_split = 1;
    // TODO: Handle case where multiple ranks are reading from a single file.
    // TODO: We could be smarter here, suppose that the number of files we end up wanting
    // is one, but each file is marginally larger than our target_rows_per_chunk, we'd end
    // up producing many small chunks.
    if (files_per_rank > 1) {
        // Figure out a guesstimated splitting.
        auto source = cudf::io::source_info(files[0]);
        auto metadata = cudf::io::read_parquet_metadata(source);
        auto const rg = metadata.rowgroup_metadata();
        auto const num_rows = metadata.num_rows();
        files_per_split =
            std::max(static_cast<int>(num_rows_per_chunk / num_rows), files_per_split);
    }
    std::vector<Node> read_tasks;
    auto options_skip_rows = options.get_skip_rows();
    auto options_num_rows =
        options.get_num_rows().value_or(std::numeric_limits<int64_t>::max());
    std::uint64_t sequence_number = 0;
    for (file_offset = 0; file_offset < files_per_rank; file_offset += files_per_split) {
        if (options_num_rows == 0) {
            break;
        }
        auto nfiles = std::min(files_per_split, files_per_rank - file_offset);
        std::vector<std::string> chunk;
        chunk.reserve(static_cast<std::size_t>(nfiles));
        std::ranges::move(
            files.begin() + file_offset,
            files.begin() + file_offset + nfiles,
            std::back_inserter(chunk)
        );
        auto source = cudf::io::source_info(std::move(chunk));
        auto metadata = cudf::io::read_parquet_metadata(source);
        auto const source_num_rows = metadata.num_rows();
        auto skip_rows = options_skip_rows;
        options_skip_rows = std::max(0L, options_skip_rows - source_num_rows);
        while (skip_rows < source_num_rows && options_num_rows > 0) {
            cudf::size_type num_rows = std::min(
                {static_cast<std::int64_t>(num_rows_per_chunk),
                 source_num_rows - skip_rows,
                 options_num_rows}
            );
            options_num_rows = std::max(0L, options_num_rows - num_rows);
            cudf::io::parquet_reader_options chunk_options{options};
            chunk_options.set_source(source);
            chunk_options.set_skip_rows(skip_rows);
            chunk_options.set_num_rows(num_rows);
            read_tasks.push_back(ctx->executor()->schedule(read_parquet_chunk(
                ctx,
                throttle,
                ctx->br()->stream_pool().get_stream(),
                std::move(chunk_options),
                // TODO: sequence number being correct relies on read_parquet_chunk
                // sending only one chunk.
                sequence_number++
            )));
            skip_rows += num_rows;
        }
    }
    co_await when_all_or_throw(std::move(read_tasks));
    co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::streaming::node
