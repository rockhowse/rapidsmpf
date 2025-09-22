/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {

Node when_all_or_throw(std::vector<Node>&& nodes) {
    auto results = co_await coro::when_all(std::move(nodes));
    // The node result itself is always `void` but we access it here to re-throw
    // possible unhandled exceptions.
    for (auto& result : results) {
        result.return_value();
    }
}

void run_streaming_pipeline(std::vector<Node> nodes) {
    coro::sync_wait(when_all_or_throw(std::move(nodes)));
}

}  // namespace rapidsmpf::streaming
