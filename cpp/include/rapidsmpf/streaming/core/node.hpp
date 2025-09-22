/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Alias for a node in a streaming pipeline.
 *
 * Nodes represent coroutine-based asynchronous operations used throughout the streaming
 * pipeline.
 */
using Node = coro::task<void>;

/**
 * @brief Produce a new coroutine that waits for completion of all tasks.
 *
 * This function schedules each node and returns a new coroutine to
 * await on after which all of them will have finished execution.
 *
 * @param nodes A vector of nodes to await.
 * @throws If any of the underlying tasks throws an exception it is
 * rethrown.
 * @return Coroutine representing the completion of all of the tasks
 * in `nodes`.
 */
Node when_all_or_throw(std::vector<Node>&& nodes);

/**
 * @brief Runs a list of nodes concurrently and waits for all to complete.
 *
 * This function schedules each node and blocks until all of them have finished execution.
 * Typically used to launch multiple producer/consumer coroutines in parallel.
 *
 * @param nodes A vector of nodes to run.
 */
void run_streaming_pipeline(std::vector<Node> nodes);

}  // namespace rapidsmpf::streaming
