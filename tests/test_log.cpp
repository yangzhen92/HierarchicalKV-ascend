/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include "debug.h"

using namespace std;
using namespace npu::hkv;

// log_format: 基础占位符替换
TEST(LogFormat, SinglePlaceholder) {
  EXPECT_EQ(log_format("n={}", 1024), "n=1024");
  EXPECT_EQ(log_format("cap={}", static_cast<size_t>(2048)), "cap=2048");
  EXPECT_EQ(log_format("flag={}", true), "flag=1");
  EXPECT_EQ(log_format("flag={}", false), "flag=0");
}

TEST(LogFormat, MultiplePlaceholders) {
  EXPECT_EQ(log_format("a={}, b={}", 1, 2), "a=1, b=2");
  EXPECT_EQ(log_format("{} {} {}", 1, 2u, 3), "1 2 3");
}

TEST(LogFormat, NoPlaceholder) {
  EXPECT_EQ(log_format("no args"), "no args");
  EXPECT_EQ(log_format("plain text"), "plain text");
}

TEST(LogFormat, EmptyFormat) { EXPECT_EQ(log_format(""), ""); }

TEST(LogFormat, PlaceholderAtEdges) {
  EXPECT_EQ(log_format("{}end", 1), "1end");
  EXPECT_EQ(log_format("start{}", 2), "start2");
  EXPECT_EQ(log_format("{}", 3), "3");
}

TEST(LogFormat, AdjacentPlaceholders) {
  EXPECT_EQ(log_format("{}{}", 1, 2), "12");
  EXPECT_EQ(log_format("{}{}{}", true, false, 7), "107");
}

// 验证各类型参数通过 operator<< 正确输出（这正是 {} 相对 printf 的优势）
TEST(LogFormat, VariousTypes) {
  EXPECT_EQ(log_format("i32={}", int32_t(-5)), "i32=-5");
  EXPECT_EQ(log_format("u32={}", uint32_t(42)), "u32=42");
  EXPECT_EQ(log_format("u64={}", uint64_t(18446744073709551615ULL)),
            "u64=18446744073709551615");
  EXPECT_EQ(log_format("f={}", 3.14), "f=3.14");
  EXPECT_EQ(log_format("s={}", string("hello")), "s=hello");
  EXPECT_EQ(log_format("c={}", 'X'), "c=X");
}

// 实际使用场景：模拟 HKV_LOG_INFO 宏的格式
TEST(LogFormat, HkvLogStyleMimic) {
  EXPECT_EQ(log_format("init begin: dim={}, init_capacity={}",
                       static_cast<size_t>(64), static_cast<size_t>(1024)),
            "init begin: dim=64, init_capacity=1024");
  EXPECT_EQ(log_format("reserve doubled: old_capacity={}, new_capacity={}",
                       static_cast<size_t>(512), static_cast<size_t>(1024)),
            "reserve doubled: old_capacity=512, new_capacity=1024");
}

// 占位符多于参数：所有 {} 在分割阶段被消费，占位符后的文字段仍会被输出
TEST(LogFormat, MorePlaceholdersThanArgs) {
  // {} 总是被当作分隔符消费，不会作为字面量保留；其后的文字段照常拼接
  EXPECT_EQ(log_format("a={}, b={}", 1), "a=1, b=");
}

// 参数多于占位符：多出的参数仍会被无分隔地拼接输出
TEST(LogFormat, MoreArgsThanPlaceholders) {
  // 注意：这不是丢弃多余参数，而是将它们紧挨着追加到末尾
  EXPECT_EQ(log_format("a={}", 1, 2, 3), "a=123");
}

// 验证 log_print 的输出格式（通过捕获 stderr）
TEST(LogPrint, OutputFormat) {
  stringstream captured;
  auto* old_buf = std::cerr.rdbuf(captured.rdbuf());
  log_print("INFO", "file.h", 42, log_format("n={}", 7));
  std::cerr.rdbuf(old_buf);
  EXPECT_EQ(captured.str(), "[INFO] file.h:42 n=7\n");
}

TEST(LogPrint, WarningAndErrorLevels) {
  {
    stringstream captured;
    auto* old_buf = std::cerr.rdbuf(captured.rdbuf());
    log_print("WARNING", "x.cpp", 10, "warn");
    std::cerr.rdbuf(old_buf);
    EXPECT_EQ(captured.str(), "[WARNING] x.cpp:10 warn\n");
  }
  {
    stringstream captured;
    auto* old_buf = std::cerr.rdbuf(captured.rdbuf());
    log_print("ERROR", "y.cpp", 20, "err");
    std::cerr.rdbuf(old_buf);
    EXPECT_EQ(captured.str(), "[ERROR] y.cpp:20 err\n");
  }
}

// 验证 HKV_LOG_INFO 宏在级别关闭时不输出
TEST(LogMacro, InfoSilentByDefault) {
  // 默认日志级别为 HKV_ERROR（2），INFO（0）不应输出
  stringstream captured;
  auto* old_buf = std::cerr.rdbuf(captured.rdbuf());
  HKV_LOG_INFO("should not appear {}", 1);
  std::cerr.rdbuf(old_buf);
  EXPECT_TRUE(captured.str().empty());
}

TEST(LogMacro, FormatWithStringLiteral) {
  // 验证宏在默认 ERROR 级别下不会触发格式化（包括字符串字面量无参调用）
  stringstream captured;
  auto* old_buf = std::cerr.rdbuf(captured.rdbuf());
  HKV_LOG_INFO("no args");
  HKV_LOG_WARNING("also no args");
  std::cerr.rdbuf(old_buf);
  EXPECT_TRUE(captured.str().empty());
}
