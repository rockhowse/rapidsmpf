/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/leaf_node.hpp>

namespace rapidsmpf::streaming::node {


Node push_to_channel(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::vector<Message> messages
) {
    // std::cout << "push_to_channel() - messages:" << messages.size() << std::endl;
    ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();

    for (auto& msg : messages) {
        RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty", std::invalid_argument);
        // std::cout << "push_to_channel() - send" << std::endl;
        co_await ch_out->send(std::move(msg));
    }
    co_await ch_out->drain(ctx->executor());
}

Node pull_from_channel(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<Message>& out_messages
) {
    // std::cout << "pull_from_channel() - out_messages" << std::endl;
    ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        out_messages.push_back(std::move(msg));
    }
}


}  // namespace rapidsmpf::streaming::node
