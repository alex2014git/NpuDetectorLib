
# Introduction

MultiNetworkPipeline is a high performance singleton thread safe module that 
allows application to infer multiple network in a pipeline, the module will take
care of network switching using latest HailoRt library taking care of performance
and new feature set advantages.  


# Usage of MultiNetworkPipeline

0. Simply include MultiNetworkPipeline.hpp and add MultiNetworkPipeline.cpp to you build.
1. GetInstance
2. InitializeHailo
3. AddNetwork (add all required network for the inference pipeline)
4. Infer (Inference for the selected network)
5. ReadOutputById (Get output result for the desired network)
6. ReleaseAllResource (on application exit or reset to start over)

NOTE: 
1. Please make sure to read .hpp file for API details.
2. For sample workflow please refer to MNP_Unittest.cpp under Test folder or the tool demo app.
3. MultiNetworkPipeline uses hpp file under Utils, just make sure it is included in you project.


# Instruction to run UnitTest (with GoogleTest framework)

1. Install GoogleTest, please follow https://github.com/google/googletest/blob/master/googletest/README.md
2. To build unittest ". build.sh"
3. run unittest ./build/x86_64/runme_test.x.x.x

