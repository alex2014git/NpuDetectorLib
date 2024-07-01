/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MultiNetworkPipeline.hpp
 * Author: Aaron
 * 
 * Created on July 5, 2021, 11:32 AM
 */

#ifndef _MultiNetworkPipeline_H_
#define _MultiNetworkPipeline_H_

#include <time.h>
#include <vector>
#include <list>
#include <ctype.h>
#include <cstring> 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <queue>
#include <hailo/hailort.h>
#include "Utils/SharedQueue.hpp"
#include "Utils/Timer.hpp"
#include "Utils/counter-master/Counter.h"
#include <type_traits>


//TODO: Implement a logging module for more flexibility
//#define LOG_INFO_ENABLE
#define LOG_WARN_ENABLE
#define LOG_ERROR_ENABLE
//#define LOG_DEBUG_ENABLE

#ifdef LOG_INFO_ENABLE 
#define DBG_INFO(MSG)   std::cout<<"[INFO] "<<MSG<<std::endl;
#else
#define DBG_INFO(MSG)
#endif

#ifdef LOG_WARN_ENABLE 
#define DBG_WARN(MSG)   std::cout<<"[WARNING] "<<MSG<<std::endl;
#else
#define DBG_WARN(MSG)
#endif

#ifdef LOG_ERROR_ENABLE 
#define DBG_ERROR(MSG)   std::cout<<"[ERROR] "<<MSG<<std::endl;
#else
#define DBG_ERROR(MSG)
#endif

#ifdef LOG_DEBUG_ENABLE 
#define DBG_DEBUG(MSG)   std::cout<<"[DEBUG] "<<MSG<<std::endl;
#else
#define DBG_DEBUG(MSG)
#endif

#define NUMBER_OF_DEV_SUPPORTED             (4)
#define TOTAL_HEF_SUPPORTED                 (10)

#define MAX_SUPPORTED_INPUT_LAYER           (4)
#define MAX_SUPPORTED_OUTPUT_LAYER          (16)


#define REQUIRE_SUCCESS_CHECK(status, label, msg) \
    do                                      \
    {                                       \
        if (HAILO_SUCCESS != (status))      \
        {                                   \
            DBG_ERROR("Hailo Code: " << status << " Msg: " << msg) \
            goto label;                     \
        }                                   \
    } while (0)



#define DEFAULT_STREAM_ID   "default"


enum class MnpReturnCode {
    
    //Success Code List
    SUCCESS                 =0,
    DUPLICATED              =1,
    NO_DATA_AVAILABLE       =2,
    
    //Failure Code list
    FAILED                  =-1,
    NOT_FOUND               =-2,
    RUNNING_INFERENCE       =-3,
    HAILO_NOT_INITIALIZED   =-4,
    INVALID_PARAMETER       =-5,
    
};


typedef struct HailoPower {
    float32_t min;
    float32_t average;
    float32_t max;
} HailoPower;


struct AtomicIntCount {
    
    Counter32   value = 0;
    Counter32   accumulated = 0;
    std::mutex  protect_mutex;
    
    bool isZeroValue() {
        std::lock_guard<std::mutex> lock(protect_mutex);  
        bool isZero = (value==0) ? true : false; 
        return isZero;
    }
    
    //Do not allow assignment
    void operator = (int) = delete;
    
    // Overload ++ when used as prefix
    void operator ++ () {
        std::lock_guard<std::mutex> lock(protect_mutex); 
        ++value;
        ++accumulated;
    }

    // Overload ++ when used as postfix
    void operator ++ (int) {
        std::lock_guard<std::mutex> lock(protect_mutex); 
        value++;
        accumulated++;

        DBG_DEBUG("AtomicInt++ = " << value.ToUnsigned() << ", Accumulated = " << accumulated.ToUnsigned());
    }

    // Overload -- when used as prefix
    void operator -- () {
        std::lock_guard<std::mutex> lock(protect_mutex); 
        --value;
    }

    // Overload -- when used as postfix
    void operator -- (int) {
        std::lock_guard<std::mutex> lock(protect_mutex); 
        value--;

        DBG_DEBUG("AtomicInt-- = " << value.ToUnsigned() << ", Accumulated = " << accumulated.ToUnsigned());        
    }
    
};


struct stNetworkModelInfo {
    
    /* MUST BE UNIQUE - Name of the network given by user */
    std::string                 id_name;

    /* Path and file name of model hef file */
    std::string                 hef_path;

    /* The Specify the order of input layer, if empty order will be default order
       NOTE:    Usually a network has only one input, in such case input_order_by_name
                can be empty. However, if network has multiple input then it is
                highly suggested to provide input_order_by_name. This given order
                will be the input_stream_index when doing inference.
    */
    std::vector<std::string>    input_order_by_name;

    /* The layer output name by order, if empty default output order will be used. */                 
    std::vector<std::string>    output_order_by_name;

    /* Default to 1 if given 0 */
    unsigned int                batch_size = 1;

    /* NOTE: Suggest to set in one of the following combination
    *  Transformation By HailoRT
    *       In this mode HailoRT library will transform the output data from UINT8 to
    *       Float32.
    *       Setting:  out_quantized=false, out_format=HAILO_FORMAT_TYPE_FLOAT32
    * 
    * Transformation By Host
    *       In this mode HailoRT library will provide output prediction result in UINT8 (native result)
    *       this is useful when Host would like to get the result in raw and filter it before 
    *       transforming it to float 32 for further processing. Its usefull to save some CPU resources
    *       for large output prediction results such as Yolov5
    *       Setting:  out_quantized=true, out_format=HAILO_FORMAT_TYPE_UINT8
    *
    */
    bool                        out_quantized = false;                  //Set to False for default
    hailo_format_type_t         out_format = HAILO_FORMAT_TYPE_FLOAT32; //Set to HAILO_FORMAT_TYPE_FLOAT32 for default    

    /* WARNING: For reserve future use, currently not supported for
     *          any other initialized value as shown below
     */
    bool                        in_quantized = false;                   //Set to False for default
    hailo_format_type_t         in_format = HAILO_FORMAT_TYPE_UINT8;    //Default to HAILO_FORMAT_TYPE_UINT8 for default
};


struct stHailoNetworkModelInfo {
    stNetworkModelInfo  NetworkModelInfo;
    hailo_hef           hefObj;
};


typedef struct qp_zp_scale_t {
    float32_t qp_zp;
    float32_t qp_scale;
} qp_zp_scale_t;


struct stHailoStreamInfo {

    uint32_t                                Device_Id;

    std::string                             NetworkIdName; //This is the same as stNetworkModelInfo's id_name
    std::string                             Stream_Id;
    hailo_configured_network_group          NetworkGroups;
    hailo_activated_network_group           NetworkGroupsActivated;
    hailo_input_vstream_params_by_name_t    NetVstreamInputParam[MAX_SUPPORTED_INPUT_LAYER];
    size_t                                  NetVstreamInputCount = 0;

    hailo_output_vstream_params_by_name_t   NetVstreamOutputParam[MAX_SUPPORTED_OUTPUT_LAYER];
    size_t                                  NetVstreamOutputCount = 0;

    hailo_input_vstream                     NetVstreamInputs[MAX_SUPPORTED_INPUT_LAYER];
    size_t                                  NetVstreamInputFrameSize[MAX_SUPPORTED_INPUT_LAYER];
    qp_zp_scale_t                           NetVstreamInputQuantInfo[MAX_SUPPORTED_INPUT_LAYER];

    hailo_output_vstream                    NetVstreamOutputs[MAX_SUPPORTED_OUTPUT_LAYER];
    size_t                                  NetVstreamOutputFrameSize[MAX_SUPPORTED_OUTPUT_LAYER];
    qp_zp_scale_t                           NetVstreamOutputQuantInfo[MAX_SUPPORTED_OUTPUT_LAYER];
    hailo_vstream_info_t                    NetVstreamOutputVstreamInfo[MAX_SUPPORTED_OUTPUT_LAYER];
    
};


/**
 * MultiNetworkPipeline is a Singleton class that defines the `GetInstance` method
 * that serves as an alternative to constructor and lets clients access the 
 * same instance of this class over and over.
 */
class MultiNetworkPipeline
{
    
    /**
     * The Singleton's constructor/destructor should always be private to
     * prevent direct construction/desctruction calls with the `new`/`delete`
     * operator.
     */
private:
                     
    static MultiNetworkPipeline *   pinstance_;
    static std::mutex               mutex_class_protection;    
    size_t                          hailo_device_found;
    hailo_device_id_t               device_ids[NUMBER_OF_DEV_SUPPORTED];
    hailo_vdevice                   vdevices[NUMBER_OF_DEV_SUPPORTED];
    stHailoNetworkModelInfo         addedNetworkModel[TOTAL_HEF_SUPPORTED];
    std::list<stHailoStreamInfo*>   streamInfoList;

private:

    int     FindAvailableHefObjSlot(void); 
    bool    isHefObjAlreadyAdded(const stNetworkModelInfo &NewNetworkInfo); 
    bool    isHefObjGivenIdDuplicated(const stNetworkModelInfo &NewNetworkInfo); 
    bool    isNetworkIdUnique(std::string network_id);
    int     FindHefObjSlot(const stNetworkModelInfo &NewNetworkInfo);
    stHailoStreamInfo*  GetNetworkStreamInfoFromMatchingNetworkId(std::string id_name);
    stHailoStreamInfo*  GetNetworkStreamInfoFromStreamChannel(std::string id_name, std::string stream_id = DEFAULT_STREAM_ID);


protected:

    MultiNetworkPipeline();
    
    ~MultiNetworkPipeline();
    
    
public:
    /**
     * MultiNetworkPipeline should not be cloneable.
     */
    MultiNetworkPipeline(MultiNetworkPipeline &other) = delete;
    
    /**
     * MultiNetworkPipeline should not be assignable.
     */
    void operator=(const MultiNetworkPipeline &) = delete;
    
    /**
     * This is the static method that controls the access to the singleton
     * instance. On the first run, it creates a singleton object and places it
     * into the static field. On subsequent runs, it returns the client existing
     * object stored in the static field.
     */
    static MultiNetworkPipeline *GetInstance();

    /**
     * Highly suggested to release all singleton resources, recommended for
     * application exits as well as reset the network to start over.
     * @return SUCCESS, RUNNING_INFERENCE
     */
    MnpReturnCode ReleaseAllResource();
    
    /**
     * Find and initialize hailo device
     * NOTE: Only one device will be initialized and used in current release
     * @return Number of Hailo Device found
     */
    size_t InitializeHailo();
    
    /**
     * Add new network, the network id_name MUST be unique  
     * @param device_id         is reserve for future use
     * @param NewNetworkInfo
     * @param stream_id         This is the unique stream_id channel that the network will be added to the pipeline
     *                          For example, if you have 4 stream, then you need to give each stream a unique stream_id
     *                          each stream is consider independent pipeline and you can add multiple network to the same
     *                          stream_id for the pipeline
     * @return SUCCESS, HAILO_NOT_INITIALIZED, RUNNING_INFERENCE, INVALID_PARAMETER
     */
    MnpReturnCode AddNetwork(uint32_t device_id, const stNetworkModelInfo &NewNetworkInfo, std::string stream_id = DEFAULT_STREAM_ID);
    
    /**
     * Release the stream channel resources
     * WARNING: It is highly recommended that you release any stream that is not longer needed so that
     *          there is available resource when you try to add network on a new stream channel (eg, new stream_id)
     * @param device_id         is reserve for future use
     * @param stream_id         This is the unique stream_id channel you want to release.
     * @return SUCCESS, HAILO_NOT_INITIALIZED, RUNNING_INFERENCE, INVALID_PARAMETER
     */
    MnpReturnCode ReleaseStreamChannel(uint32_t device_id, std::string stream_id = DEFAULT_STREAM_ID);

    /**
     * WARNING: Calling this method require to switch the network (if current 
     *          active network is not the same) before we can obtain the 
     *          input size of the network stream. This is NOT recommended
     *          to be call often, if require try to call for all added network
     *          before inference.
     * 
     * @param [in]  id_name, the network id name to get the input size
     * @param [out] NetworkInputSize, the input size of the network 
     * @param [in]  input_stream_index, this is the input index base on the order given from AddNetwork
     *                                  NewNetworkInfo.input_order_by_name.
     *                                  If network only contain one input then this can be ignored as by default
     *                                  its set to 0 (first input stream).
     * 
     * @return SUCCESS, NOT_FOUND, FAILED, HAILO_NOT_INITIALIZED
     */
    MnpReturnCode GetNetworkInputSize(const std::string &id_name, size_t &NetworkInputSize, size_t input_stream_index = 0);
    
    /**
     * This is require if output format is not translated (eg, not quantized and UINT8 format) which
     * host is require to transfor it to float 32 for further post processing. 
     * 
     * WARNING: Calling this method require to switch the network (if current 
     *          active network is not the same) before we can obtain the 
     *          network output quantization info from the network stream. This is NOT recommended
     *          to be call often, if require try to call for all added network
     *          before inference.
     * 
     * @param [in]  id_name, the network id name to get the input size
     * @param [out] NetworkQuantInfo, the quatization info in order of output_order_by_name given when adding the network.
     *                                if output_order_by_name is not given then the order is default from network.  
     * @param [in]  get_from_output_stream, the quantization from output or input stream, 
     *                                      true to get from output stream and false to get input stream quantization info, 
     *                                      default to output stream
     * @return SUCCESS, NOT_FOUND, FAILED, HAILO_NOT_INITIALIZED
     */
    MnpReturnCode GetNetworkQuantizationInfo(const std::string &id_name, std::vector<qp_zp_scale_t> &NetworkQuantInfo, bool get_from_output_stream = true);

    MnpReturnCode GetNetworkVstream_Info(const std::string &id_name, std::vector<hailo_vstream_info_t> &NetworkVstreamInfo, bool get_from_output_stream = true);

    /**
     * Infer data to specified network by id_name, data will be queued and is
     * FIFO.
     *
     * WARNING: When batch size is greater than 1, it is application responsibility that 
     *          data infer is continuos and sequential to multiple of the network batch size
     *          For example, if 2 network, A with batch size 2, and B with batch size 1
     *          make sure when infer network A, data infer is continuos before infer network B
     *          such as
     *              "infer A frame 1" -> "infer A frame 2" -> "infer B frame 1" 
     *          Bad example will be
     *              "infer A frame 1" -> "infer B frame 1" -> "infer A frame 2"
     *          With Bad example above the infer B data frame WILL be dropped! 
     * 
     * @param id_name           the network id name to infer
     * @param data              the data to infer.
     * @param stream_id         This is the unique stream_id that you want to infer the network that is already added
     *                          to the pipeline of this stream when calling AddNetwork
     * 
     * @param input_stream_index, this is the input index base on the order given from AddNetwork
     *                            NewNetworkInfo.input_order_by_name.
     *                            If network only contain one input then this can be ignored as by default
     *                            its set to 0 (first input stream).
     * @return SUCCESS, NOT_FOUND, FAILED, HAILO_NOT_INITIALIZED
     */
    MnpReturnCode Infer(const std::string &id_name, const std::vector<uint8_t> &data, std::string stream_id = DEFAULT_STREAM_ID, size_t input_stream_index = 0);


    /**
     * Get the output data by network name
     * @param id_name           the network id name to read the output prediction
     * @param output_buffer     all output layer in a vector/list in sequence given by
     *                          output_order_by_name from network model info.
     * @param stream_id         this is the unique stream_id that you want to get the prediction output
     * @return SUCCESS, NOT_FOUND, NO_DATA_AVAILABLE
     */
    MnpReturnCode ReadOutputById(const std::string &id_name, std::vector<std::vector<float32_t>>& output_buffer, std::string stream_id = DEFAULT_STREAM_ID);

    MnpReturnCode ReadOutputById(const std::string &id_name, std::vector<std::vector<uint8_t>>& output_buffer, std::string stream_id = DEFAULT_STREAM_ID);


    /**
     * Prepare the size of the output buffer that is going to be used for ReadOutputById
     * NOTE:    after calling this function output_buffer is reusable, this API simply resize the given output_buffer
     *          there is no need to call this API everytime before you call ReadOutputById.
     * @param id_name           the network id name to prepare the output buffer
     * @param output_buffer     Provide the output buffer vector, it will resize based on the network output and prepare
     *                          it so that it can be used when calling ReadOutputById
     * @param stream_id         this is the unique stream_id.
     * @return SUCCESS, NOT_FOUND, HAILO_NOT_INITIALIZED
     */
    template<typename T>
    MnpReturnCode InitializeOutputBuffer(const std::string &id_name, std::vector<std::vector<T>>& output_buffer, std::string stream_id = DEFAULT_STREAM_ID)
    {
        MnpReturnCode RetCode = MnpReturnCode::NOT_FOUND;
        stHailoStreamInfo* pStreamInfo = nullptr;

        if (!hailo_device_found)
            return MnpReturnCode::HAILO_NOT_INITIALIZED;

        mutex_class_protection.lock();

        pStreamInfo = MultiNetworkPipeline::GetNetworkStreamInfoFromStreamChannel(id_name, stream_id);

        mutex_class_protection.unlock();

        if (pStreamInfo) {
            output_buffer.resize(pStreamInfo->NetVstreamOutputCount);
            for (size_t i = 0; i < pStreamInfo->NetVstreamOutputCount; i++) {
                output_buffer[i].resize(pStreamInfo->NetVstreamOutputFrameSize[i]/sizeof(T));
            }

            RetCode = MnpReturnCode::SUCCESS;
        }

        return RetCode;
    }


private:

    template<typename T>
    MnpReturnCode CheckOutputBuffer(stHailoStreamInfo* pStreamInfo, std::vector<std::vector<T>>& output_buffer)
    {
        MnpReturnCode RetCode = MnpReturnCode::INVALID_PARAMETER;

        if (pStreamInfo) {

            if (output_buffer.size() != pStreamInfo->NetVstreamOutputCount)
                return RetCode;

            output_buffer.resize(pStreamInfo->NetVstreamOutputCount);
            for (size_t i = 0; i < pStreamInfo->NetVstreamOutputCount; i++) {

                if (output_buffer[i].size() != pStreamInfo->NetVstreamOutputFrameSize[i]/sizeof(T))
                    return RetCode;
            }

            RetCode = MnpReturnCode::SUCCESS;
        }

        return RetCode;
    }

};



#endif //_MultiNetworkPipeline_H_

