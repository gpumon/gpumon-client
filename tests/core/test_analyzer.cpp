#include <gtest/gtest.h>
#include "gpufl/gpufl.hpp"

TEST(CoreLogic, InitOptionsDefault) {
    gpufl::InitOptions opts;
    EXPECT_EQ(opts.appName, "gpufl");
    EXPECT_FALSE(opts.samplingAutoStart);
    EXPECT_TRUE(opts.enableProfiling);
}

TEST(CoreLogic, BackendKindEnum) {
    EXPECT_EQ(static_cast<int>(gpufl::BackendKind::Auto), 0);
    EXPECT_EQ(static_cast<int>(gpufl::BackendKind::Nvidia), 1);
}
