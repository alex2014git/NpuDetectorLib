# Unit Tests (Reserved for Phase 2)

This directory is reserved for true unit tests that don't require:
- HailoRT library
- Hailo NPU hardware
- Actual model files (HEF)

Unit tests will be possible after Phase 2 splits the library into:
- `PipelineCore` - Graph, Context, Scheduler (no Hailo)
- `NpuBackend` - Backend abstraction
- `HailoBackend` - Hailo-specific implementation

Current tests are in ../hardware/ since they all require Hailo hardware.
