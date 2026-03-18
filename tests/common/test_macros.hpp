#ifndef TEST_MACROS_HPP
#define TEST_MACROS_HPP

#include <iostream>
#include <cmath>

// Test counters - define in each test's main()
extern int g_tests_passed;
extern int g_tests_failed;

// Basic assertion
#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

// Assertion with message
#define TEST_ASSERT_MSG(condition, msg) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

// Float comparison
#define TEST_ASSERT_FLOAT_EQ(a, b, eps) \
    do { \
        if (std::fabs((a) - (b)) > (eps)) { \
            std::cerr << "FAIL: " << #a << " != " << #b << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

// Print summary
#define TEST_PRINT_SUMMARY(name) \
    do { \
        std::cout << "========================================" << std::endl; \
        std::cout << name << " Summary" << std::endl; \
        std::cout << "========================================" << std::endl; \
        std::cout << "Passed: " << g_tests_passed << std::endl; \
        std::cout << "Failed: " << g_tests_failed << std::endl; \
        std::cout << (g_tests_failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl; \
    } while(0)

#endif // TEST_MACROS_HPP
