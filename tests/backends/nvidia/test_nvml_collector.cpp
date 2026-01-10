#include <gtest/gtest.h>
#include "common/test_utils.hpp"
#include "gpufl/backends/nvidia/nvml_collector.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML

class NvmlCollectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_CUDA();
        std::string reason;
        if (!gpufl::nvidia::NvmlCollector::isAvailable(&reason)) {
            GTEST_SKIP() << "NVML not available: " << reason;
        }
    }
};

TEST_F(NvmlCollectorTest, Availability) {
    // Already checked in SetUp, but good to have a dedicated test
    EXPECT_TRUE(gpufl::nvidia::NvmlCollector::isAvailable());
}

TEST_F(NvmlCollectorTest, SampleDynamicMetrics) {
    gpufl::nvidia::NvmlCollector collector;
    auto samples = collector.sampleAll();

    EXPECT_FALSE(samples.empty());

    for (const auto& sample : samples) {
        EXPECT_GE(sample.deviceId, 0);
        EXPECT_FALSE(sample.name.empty());
        
        // Memory metrics
        EXPECT_GT(sample.totalMiB, 0);
        EXPECT_LE(sample.usedMiB, sample.totalMiB);
        // Sum might be off by 1 MiB due to floor division in toMiB(bytes)
        EXPECT_NEAR(sample.totalMiB, sample.usedMiB + sample.freeMiB, 1.1);

        // Utilization metrics (0-100)
        EXPECT_LE(sample.gpuUtil, 100);
        EXPECT_LE(sample.memUtil, 100);

        // Clock metrics
        EXPECT_GT(sample.clockSm, 0);
        EXPECT_GT(sample.clockMem, 0);
        
        // Temperature
        EXPECT_GT(sample.tempC, 0);
        EXPECT_LT(sample.tempC, 120); // Sanity check
    }
}

#endif
