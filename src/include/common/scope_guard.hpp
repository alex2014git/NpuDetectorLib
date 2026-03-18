#ifndef SCOPE_GUARD_HPP
#define SCOPE_GUARD_HPP

#include <functional>
#include <utility>

namespace npu {

// Simple scope guard for RAII cleanup
class ScopeGuard {
public:
    explicit ScopeGuard(std::function<void()> cleanup)
        : cleanup_(std::move(cleanup)), active_(true) {}

    ~ScopeGuard() {
        if (active_ && cleanup_) {
            cleanup_();
        }
    }

    // Disable copy
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    // Enable move
    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup_(std::move(other.cleanup_)), active_(other.active_) {
        other.active_ = false;
    }

    void dismiss() { active_ = false; }

private:
    std::function<void()> cleanup_;
    bool active_;
};

// Helper macro for common use case
#define NPU_SCOPE_GUARD(name, cleanup) \
    npu::ScopeGuard name(cleanup)

} // namespace npu

#endif // SCOPE_GUARD_HPP
