/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MultiNetworkPipeline.cpp
 * Author: Aaron
 * 
 * Created on July 5, 2021, 11:32 AM
 */

#include "MultiNetworkPipeline/MultiNetworkPipeline.hpp"


/**
 * Static methods should be defined outside the class.
 */

MultiNetworkPipeline* MultiNetworkPipeline::pinstance_{nullptr};
std::mutex MultiNetworkPipeline::mutex_class_protection;


/**
 * Internal Helper Functions
 */

void SplitStringBy(std::string str, std::string splitBy, std::vector<std::string>& tokens)
{
    /* Store the original string in the array, so we can loop the rest
     * of the algorithm. */
    tokens.push_back(str);

    // Store the split index in a 'size_t' (unsigned integer) type.
    size_t splitAt;
    // Store the size of what we're splicing out.
    size_t splitLen = splitBy.size();
    // Create a string for temporarily storing the fragment we're processing.
    std::string frag;
    // Loop infinitely - break is internal.
    while(true)
    {
        /* Store the last string in the vector, which is the only logical
         * candidate for processing. */
        frag = tokens.back();
        /* The index where the split is. */
        splitAt = frag.find(splitBy);
        // If we didn't find a new split point...
        if(splitAt == std::string::npos)
        {
            // Break the loop and (implicitly) return.
            break;
        }
        /* Put everything from the left side of the split where the string
         * being processed used to be. */
        tokens.back() = frag.substr(0, splitAt);
        /* Push everything from the right side of the split to the next empty
         * index in the vector. */
        tokens.push_back(frag.substr(splitAt+splitLen, frag.size()-(splitAt+splitLen)));
    }
}



/*
 * PRIVATE METHOD
 */

int MultiNetworkPipeline::FindAvailableHefObjSlot() 
{

    int HefObjSlot = -1;
    for (int k = 0; k < TOTAL_HEF_SUPPORTED; k++) {
        if (addedNetworkModel[k].hefObj == NULL) {
            HefObjSlot = k;
            break;
        }
    }
        
    return HefObjSlot;
}


bool MultiNetworkPipeline::isHefObjGivenIdDuplicated(const stNetworkModelInfo &NewNetworkInfo)
{
    bool IdDuplicated = false;
    for (int k = 0; k < TOTAL_HEF_SUPPORTED; k++) {
        if (addedNetworkModel[k].hefObj != NULL) {

            if (addedNetworkModel[k].NetworkModelInfo.id_name.compare(NewNetworkInfo.id_name) == 0) {
                IdDuplicated = true;
                break;
            }
        }
    }

    return IdDuplicated;
}


bool MultiNetworkPipeline::isNetworkIdUnique(std::string network_id)
{
    bool NetworkIdIsUnique = true;

    std::list <stHailoStreamInfo*> :: iterator itr;
    for(itr = streamInfoList.begin(); itr != streamInfoList.end(); itr++) {
        if ((*itr)->NetworkIdName.compare(network_id) == 0) {
            NetworkIdIsUnique = false;
            break;
        }
    }

    return NetworkIdIsUnique;
}


int MultiNetworkPipeline::FindHefObjSlot(const stNetworkModelInfo &NewNetworkInfo)
{
    int HefObjSlot = -1;
    for (int k = 0; k < TOTAL_HEF_SUPPORTED; k++) {
        if (addedNetworkModel[k].hefObj != NULL) {

            if (addedNetworkModel[k].NetworkModelInfo.hef_path.compare(NewNetworkInfo.hef_path) == 0) {
                HefObjSlot = k;
                break;
            }
        }
    }

    return HefObjSlot;
}


bool MultiNetworkPipeline::isHefObjAlreadyAdded(const stNetworkModelInfo &NewNetworkInfo)
{
    bool AlreadyAdded = false;
    for (int k = 0; k < TOTAL_HEF_SUPPORTED; k++) {
        if (addedNetworkModel[k].hefObj != NULL) {

            if (addedNetworkModel[k].NetworkModelInfo.hef_path.compare(NewNetworkInfo.hef_path) == 0) {
                AlreadyAdded = true;
                break;
            }
        }
    }

    return AlreadyAdded;
}


stHailoStreamInfo*  MultiNetworkPipeline::GetNetworkStreamInfoFromMatchingNetworkId(std::string id_name)
{
    stHailoStreamInfo* pHailoStreamInfo = nullptr;

    std::list <stHailoStreamInfo*> :: iterator itr;
    for(itr = streamInfoList.begin(); itr != streamInfoList.end(); itr++) {
        if ((*itr)->NetworkIdName.compare(id_name) == 0) {
            pHailoStreamInfo = *itr;
            break;
        }
    }

    return pHailoStreamInfo;
}


stHailoStreamInfo*  MultiNetworkPipeline::GetNetworkStreamInfoFromStreamChannel(std::string id_name, std::string stream_id /*= "default" */)
{

    stHailoStreamInfo* pHailoStreamInfo = nullptr;

    std::list <stHailoStreamInfo*> :: iterator itr;
    for(itr = streamInfoList.begin(); itr != streamInfoList.end(); itr++) {

        if ((*itr)->Stream_Id.compare(stream_id) != 0)
            continue;

        if ((*itr)->NetworkIdName.compare(id_name) == 0) {
            pHailoStreamInfo = *itr;
            break;
        }
    }

    return pHailoStreamInfo;

}



/*
 * PUBLIC METHOD
 */


MultiNetworkPipeline::MultiNetworkPipeline()
{

#ifdef LARGE_INFER_QUEUE_FOR_UNIT_TEST
    DBG_WARN("LARGE_INFER_QUEUE_FOR_UNIT_TEST is defined, please remove from CMakeList if this is not a unit test build");
#endif

    hailo_device_found = 0;
            
    if (InitializeHailo() == 0) {
        DBG_WARN("No Hailo Device Found");
    }
    
}

MultiNetworkPipeline::~MultiNetworkPipeline()
{
    
}

/* PUBLIC METHOD */


/**
 * The first time we call GetInstance we will lock the storage location
 *      and then we make sure again that the variable is null and then we
 *      set the value. RU:
 */
MultiNetworkPipeline *MultiNetworkPipeline::GetInstance()
{
    // mutex_class_protection is automatically released when lock
    // goes out of scope
    std::lock_guard<std::mutex> lock(mutex_class_protection);
    if (pinstance_ == nullptr)
    {
        pinstance_ = new MultiNetworkPipeline();
    }
    return pinstance_;
}


MnpReturnCode MultiNetworkPipeline::ReleaseStreamChannel(uint32_t device_id, std::string stream_id /*= "default"*/)
{

    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;

    if (device_id >= hailo_device_found)
    {
        DBG_WARN("device id exceeded current maximum hailodevice found (" << hailo_device_found << ")");
        return MnpReturnCode::INVALID_PARAMETER;
    }

    //Protect resource
    std::lock_guard<std::mutex> lock(mutex_class_protection);

    std::list <stHailoStreamInfo*> :: iterator itr;
    for(itr = streamInfoList.begin(); itr != streamInfoList.end(); itr++) {

        if ((*itr)->Device_Id != device_id)
            continue;

        if ((*itr)->Stream_Id.compare(stream_id) == 0) {

            (void)hailo_release_output_vstreams((*itr)->NetVstreamOutputs,
                                                (*itr)->NetVstreamOutputCount);

            (void)hailo_release_input_vstreams( (*itr)->NetVstreamInputs,
                                                (*itr)->NetVstreamInputCount);

            streamInfoList.erase(itr--);
        }

    }

    return MnpReturnCode::SUCCESS;
}


MnpReturnCode MultiNetworkPipeline::ReleaseAllResource()
{
    MnpReturnCode RetCode = MnpReturnCode::SUCCESS;        
    
    //Protect resource
    std::lock_guard<std::mutex> lock(mutex_class_protection);

    std::list <stHailoStreamInfo*> :: iterator itr;
    for(itr = streamInfoList.begin(); itr != streamInfoList.end(); itr++) {

        (void)hailo_release_output_vstreams((*itr)->NetVstreamOutputs,
                                            (*itr)->NetVstreamOutputCount);

        (void)hailo_release_input_vstreams( (*itr)->NetVstreamInputs,
                                            (*itr)->NetVstreamInputCount);

        streamInfoList.erase(itr--);

    }

    for (int j = 0; j < TOTAL_HEF_SUPPORTED; j++) {     
        if (addedNetworkModel[j].hefObj != NULL) {
            (void) hailo_release_hef(addedNetworkModel[j].hefObj);
        }
    }


    for (int i = 0; i < NUMBER_OF_DEV_SUPPORTED; i++) {
        if (vdevices[i] != NULL) {
            //std::cout << "Release Device-" << i << "Address: " << &vdevices[i] << std::endl;
            (void) hailo_release_vdevice(vdevices[i]);
            
        }        
    }

    hailo_device_found = 0;
    
    //std::cout << "Release Done" << std::endl;
    return RetCode;
}


size_t MultiNetworkPipeline::InitializeHailo()
{    
    hailo_status status = HAILO_SUCCESS;

    if (hailo_device_found)
        goto init_exit;
    
    for (int i = 0; i < NUMBER_OF_DEV_SUPPORTED; i++) {
        vdevices[i] = NULL;        
    }

    for (int j = 0; j < TOTAL_HEF_SUPPORTED; j++) {        
        addedNetworkModel[j].hefObj = NULL;
    }

    try {

        hailo_device_found = NUMBER_OF_DEV_SUPPORTED;
        hailo_scan_devices(NULL, device_ids, &hailo_device_found);
        
        if (hailo_device_found > NUMBER_OF_DEV_SUPPORTED)
        {
            DBG_ERROR("-E- found " << hailo_device_found << " device(s) but exceeding max device support of " << NUMBER_OF_DEV_SUPPORTED);
            hailo_device_found = 0;
        }

        for (size_t i = 0; i < hailo_device_found; i++) {
            hailo_vdevice_params_t params = {0};

            if (hailo_init_vdevice_params(&params) !=  HAILO_SUCCESS){
                DBG_ERROR("-E- failed to create hailo_init_vdevice_params");
                break;
            }

            //TODO: Here we create each device as individual entity, application can have the flexibility
            //      to decide multinetwork schedule for each hailo device, in the future we might be able to
            //      support multi-device network scheduler, when it does we will have to revice the code
            //      here and make sure we can have the option for it.
            params.device_count = 1;
            params.device_ids = &device_ids[i];
            params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
            params.multi_process_service = false;       //Set default to false, we can try to support this later
            status = hailo_create_vdevice(&params, &vdevices[i]);
            
            //std::cout << "Allocated vDevice Address: " << &vdevices[i] << std::endl;

            REQUIRE_SUCCESS_CHECK(status, init_exit, "Failed to create hailo_create_vdevice");

        }

       
    } catch (std::exception const& e) {
        DBG_ERROR("-E- create device failed" << e.what());
    }

    
init_exit:
    
    return hailo_device_found;
}



MnpReturnCode MultiNetworkPipeline::AddNetwork(uint32_t device_id, const stNetworkModelInfo &NewNetworkInfo, std::string stream_id /* = "default" */)
{
    
    MnpReturnCode RetCode = MnpReturnCode::FAILED;
    hailo_status status = HAILO_SUCCESS;
    stHailoStreamInfo* pHailoStreamInfoObj = nullptr;

    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;

    if (device_id >= hailo_device_found)
    {
        DBG_WARN("device id exceeded current maximum hailodevice found (" << hailo_device_found << ")");
        return MnpReturnCode::INVALID_PARAMETER;        
    }

    if ((NewNetworkInfo.out_format != HAILO_FORMAT_TYPE_UINT8) && 
        (NewNetworkInfo.out_format != HAILO_FORMAT_TYPE_FLOAT32))
    {
        DBG_WARN("Configured output format not yet supported.");
        return MnpReturnCode::INVALID_PARAMETER;
    }

    //Any manipulation method we lock the resources first
    std::lock_guard<std::mutex> lock(mutex_class_protection);            
  
    /* Create HEF Obj */

    if (isHefObjAlreadyAdded(NewNetworkInfo) == false) {

        if (isHefObjGivenIdDuplicated(NewNetworkInfo)) {
            DBG_WARN("Add new network given id_name already exist (must be unique) : " << NewNetworkInfo.id_name);
            return MnpReturnCode::INVALID_PARAMETER;
        }

        int AvailableSLot = FindAvailableHefObjSlot();
        if (AvailableSLot < 0) {
            DBG_WARN("Max Hef allowed reached (Please increase TOTAL_HEF_SUPPORTED define).");
            return MnpReturnCode::INVALID_PARAMETER;
        }

        //Add the new network to the list
        //NOTE: Each network is only required to be added once (eg, hailo_create_hef_file). Creating same hef file MULTIPLE TIMES
        //      will waste resources (memory), although doing it will still work but its waste of memory especially in enbedded system with
        //      limited memory.

        addedNetworkModel[AvailableSLot].NetworkModelInfo = NewNetworkInfo;

        status = hailo_create_hef_file(&addedNetworkModel[AvailableSLot].hefObj, NewNetworkInfo.hef_path.c_str());
        REQUIRE_SUCCESS_CHECK(status, l_exit, "Failed to create hef file");

    }

    /* Config vdevice (network group)*/
    {
        int HefObjSlot = FindHefObjSlot(NewNetworkInfo);
        if (HefObjSlot < 0) {
            DBG_WARN("Unable to find matched hef path (network) which should be already added in the list");
            return MnpReturnCode::INVALID_PARAMETER;
        }

        if (isNetworkIdUnique(NewNetworkInfo.id_name) == false) {
            DBG_WARN("Give network_id MUST be unique across all devices & streams");
            return MnpReturnCode::INVALID_PARAMETER;
        }


        hailo_configure_params_t configure_params = {0};
        size_t network_groups_size = 1;

        status = hailo_init_configure_params(addedNetworkModel[HefObjSlot].hefObj, HAILO_STREAM_INTERFACE_PCIE, &configure_params);
        REQUIRE_SUCCESS_CHECK(status, l_exit, "Failed to hailo_init_configure_params");

        // Allocate new stream object
        pHailoStreamInfoObj = new stHailoStreamInfo();

        pHailoStreamInfoObj->Device_Id = device_id;
        status = hailo_configure_vdevice(   vdevices[device_id], 
                                            addedNetworkModel[HefObjSlot].hefObj, 
                                            &configure_params,
                                            &pHailoStreamInfoObj->NetworkGroups,
                                            &network_groups_size);
        REQUIRE_SUCCESS_CHECK(status, l_release, "Failed to hailo_configure_vdevice");
        if (network_groups_size > 1) {
            DBG_WARN("Hef with more that one network group, currently not supported. Contact field support agent for detail");
        }

        pHailoStreamInfoObj->NetworkIdName = NewNetworkInfo.id_name;
        pHailoStreamInfoObj->Stream_Id = stream_id;

        /* Make Input/Output VStream */
        pHailoStreamInfoObj->NetVstreamInputCount = MAX_SUPPORTED_INPUT_LAYER;
        status = hailo_make_input_vstream_params(pHailoStreamInfoObj->NetworkGroups,
                                                 NewNetworkInfo.in_quantized,
                                                 NewNetworkInfo.in_format,
                                                 pHailoStreamInfoObj->NetVstreamInputParam,
                                                 &pHailoStreamInfoObj->NetVstreamInputCount);

        REQUIRE_SUCCESS_CHECK(status, l_release, "Failed making input virtual stream params");

        pHailoStreamInfoObj->NetVstreamOutputCount = MAX_SUPPORTED_OUTPUT_LAYER;

        status = hailo_make_output_vstream_params(  pHailoStreamInfoObj->NetworkGroups,
                                                    NewNetworkInfo.out_quantized, 
                                                    NewNetworkInfo.out_format,
                                                    pHailoStreamInfoObj->NetVstreamOutputParam,
                                                    &pHailoStreamInfoObj->NetVstreamOutputCount);
        
        REQUIRE_SUCCESS_CHECK(status, l_release, "Failed making output virtual stream params");
    
    }

    /* Create Virtual Input Stream */
    {
        bool bUseDefaultInputOrder = true;
        if (!NewNetworkInfo.input_order_by_name.empty())
            bUseDefaultInputOrder = false;

        size_t InputStreamCount = pHailoStreamInfoObj->NetVstreamInputCount;
        for (size_t i = 0; i < InputStreamCount; i++)
        {

            if (bUseDefaultInputOrder)
            {

//#ifdef LARGE_INFER_QUEUE_FOR_UNIT_TEST
                //pHailoStreamInfoObj->NetVstreamInputParam[i].params.queue_size = 30;
//#endif
                status = hailo_create_input_vstreams(   pHailoStreamInfoObj->NetworkGroups,
                                                        &pHailoStreamInfoObj->NetVstreamInputParam[i],
                                                        1, 
                                                        &pHailoStreamInfoObj->NetVstreamInputs[i]);
                REQUIRE_SUCCESS_CHECK(status, l_release, "Failed creating virtual input stream");
                DBG_INFO( "inut_stream_by_name " << pHailoStreamInfoObj->NetVstreamInputParam[i].name);

            }
            else 
            {
                bool bFound = false;
                for (size_t j = 0; j < InputStreamCount; j++)
                {
                    if (strcmp(pHailoStreamInfoObj->NetVstreamInputParam[j].name, NewNetworkInfo.input_order_by_name[i].c_str()) == 0)
                    {
                        status = hailo_create_input_vstreams(   pHailoStreamInfoObj->NetworkGroups,
                                                                &pHailoStreamInfoObj->NetVstreamInputParam[j],
                                                                1, 
                                                                &pHailoStreamInfoObj->NetVstreamInputs[i]);

                        DBG_INFO( "inut_stream_by_name " << pHailoStreamInfoObj->NetVstreamInputParam[j].name);

                        REQUIRE_SUCCESS_CHECK(status, l_release, "Failed creating virtual input stream");

                        bFound = true;
                        break;       
                    }
                }

                if (bFound == false)
                {
                    DBG_WARN("Unable to find given input layer name " << NewNetworkInfo.input_order_by_name[i]);
                    goto l_release;
                }             
            }


            status = hailo_get_input_vstream_frame_size(pHailoStreamInfoObj->NetVstreamInputs[i],
                                                        &pHailoStreamInfoObj->NetVstreamInputFrameSize[i]);
            REQUIRE_SUCCESS_CHECK(status, l_release, "Failed getting input virtual stream frame size");

            //Get input stream info
            hailo_vstream_info_t in_vstream_info;
            status = hailo_get_input_vstream_info(  pHailoStreamInfoObj->NetVstreamInputs[i],
                                                    &in_vstream_info);
            REQUIRE_SUCCESS_CHECK(status, l_release, "Failed hailo_get_input_vstream_info");
            
            /* Get the quantization info */
            qp_zp_scale_t transformScale = {in_vstream_info.quant_info.qp_zp,
                                            in_vstream_info.quant_info.qp_scale};
            pHailoStreamInfoObj->NetVstreamInputQuantInfo[i] = transformScale;
            //std::cout << "input(" <<  in_vstream_info.name << ")stream format type: " << in_vstream_info.format.type << ", quant info scale = " << transformScale.qp_scale << " zero point = " << transformScale.qp_zp << std::endl;

        }
    }

    /* Create Virtual Output Stream */
    {
        bool bUseDefaultOutputOrder = true;
        if (!NewNetworkInfo.output_order_by_name.empty())
            bUseDefaultOutputOrder = false;

        size_t OutputStreamCount = pHailoStreamInfoObj->NetVstreamOutputCount;

        for (size_t i = 0; i < OutputStreamCount; i++)
        {

            //pHailoStreamInfoObj->NetVstreamOutputParam[i].params.queue_size = 30;
            if (bUseDefaultOutputOrder)
            {
                status = hailo_create_output_vstreams(  pHailoStreamInfoObj->NetworkGroups,
                                                        &pHailoStreamInfoObj->NetVstreamOutputParam[i],
                                                        1, 
                                                        &pHailoStreamInfoObj->NetVstreamOutputs[i]);
                REQUIRE_SUCCESS_CHECK(status, l_release, "Failed creating virtual output stream");

                DBG_INFO( "output_stream_by_name " << pHailoStreamInfoObj->NetVstreamOutputParam[i].name);

            }
            else 
            {
                bool bFound = false;
                for (size_t j = 0; j < OutputStreamCount; j++)
                {
                    if (strcmp(pHailoStreamInfoObj->NetVstreamOutputParam[j].name, NewNetworkInfo.output_order_by_name[i].c_str()) == 0)
                    {
                        status = hailo_create_output_vstreams(  pHailoStreamInfoObj->NetworkGroups,
                                                                &pHailoStreamInfoObj->NetVstreamOutputParam[j],
                                                                1, 
                                                                &pHailoStreamInfoObj->NetVstreamOutputs[i]);

                        DBG_INFO( "output_stream_by_name " << pHailoStreamInfoObj->NetVstreamOutputParam[j].name);

                        REQUIRE_SUCCESS_CHECK(status, l_release, "Failed creating virtual output stream");


                        bFound = true;
                        break;       
                    }
                }

                if (bFound == false)
                {
                    DBG_WARN("Unable to find given output layer name " << NewNetworkInfo.output_order_by_name[i]);
                    goto l_exit;
                }             
            }


            status = hailo_get_output_vstream_frame_size(pHailoStreamInfoObj->NetVstreamOutputs[i],
                                                        &pHailoStreamInfoObj->NetVstreamOutputFrameSize[i]);
            REQUIRE_SUCCESS_CHECK(status, l_release, "Failed getting output virtual stream frame size");

            //Get output stream info
            hailo_vstream_info_t out_vstream_info;
            status = hailo_get_output_vstream_info( pHailoStreamInfoObj->NetVstreamOutputs[i],
                                                    &out_vstream_info);
            REQUIRE_SUCCESS_CHECK(status, l_release, "Failed hailo_get_output_vstream_info");

            /* Get the quantization info */
            qp_zp_scale_t transformScale = {out_vstream_info.quant_info.qp_zp,
                                            out_vstream_info.quant_info.qp_scale};
            pHailoStreamInfoObj->NetVstreamOutputQuantInfo[i] = transformScale;
            pHailoStreamInfoObj->NetVstreamOutputVstreamInfo[i] = out_vstream_info;
            //std::cout << "output(" <<  out_vstream_info.name << ")stream format type: " << out_vstream_info.format.type << ", quant info scale = " << transformScale.qp_scale << " zero point = " << transformScale.qp_zp << std::endl;

        }
    }

    // Add the newly added network on device/stream to the list.
    streamInfoList.push_back(pHailoStreamInfoObj);

    RetCode = MnpReturnCode::SUCCESS;

    goto l_exit;

l_release:

    if (pHailoStreamInfoObj)
        delete pHailoStreamInfoObj;


l_exit:

    return RetCode;
}

MnpReturnCode MultiNetworkPipeline::GetNetworkInputSize(const std::string &id_name, size_t &NetworkInputSize, size_t input_stream_index /*= 0*/)
{
    MnpReturnCode RetCode = MnpReturnCode::NOT_FOUND;
    NetworkInputSize = 0;
    
    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;    

    //Protect resource
    std::lock_guard<std::mutex> lock(mutex_class_protection);

    stHailoStreamInfo* pStreamInfo = MultiNetworkPipeline::GetNetworkStreamInfoFromMatchingNetworkId(id_name);

    if (pStreamInfo) {

        if (input_stream_index >= pStreamInfo->NetVstreamInputCount) {
            RetCode = MnpReturnCode::INVALID_PARAMETER;
        }
        else {
            NetworkInputSize = pStreamInfo->NetVstreamInputFrameSize[input_stream_index];
            RetCode = MnpReturnCode::SUCCESS;
        }
    }

    return RetCode;
}


MnpReturnCode MultiNetworkPipeline::GetNetworkQuantizationInfo(const std::string &id_name, std::vector<qp_zp_scale_t> &NetworkQuantInfo, bool get_from_output_stream /* = true */)
{
    MnpReturnCode RetCode = MnpReturnCode::NOT_FOUND;
    
    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;    

    //Protect resource
    std::lock_guard<std::mutex> lock(mutex_class_protection);

    stHailoStreamInfo* pStreamInfo = MultiNetworkPipeline::GetNetworkStreamInfoFromMatchingNetworkId(id_name);

    if (pStreamInfo) {

        if (get_from_output_stream) {

            for (size_t i = 0; i < pStreamInfo->NetVstreamOutputCount; i++) {
                NetworkQuantInfo.push_back(pStreamInfo->NetVstreamOutputQuantInfo[i]);
            }
        }
        else {

            for (size_t i = 0; i < pStreamInfo->NetVstreamInputCount; i++) {
                NetworkQuantInfo.push_back(pStreamInfo->NetVstreamInputQuantInfo[i]);
            }

        }

        RetCode = MnpReturnCode::SUCCESS;
    }
    
    return RetCode;
}

MnpReturnCode MultiNetworkPipeline::GetNetworkVstream_Info(const std::string &id_name, std::vector<hailo_vstream_info_t> &NetworkVstreamInfo, bool get_from_output_stream /* = true */)
{
    MnpReturnCode RetCode = MnpReturnCode::NOT_FOUND;
    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;

    //Protect resource
    std::lock_guard<std::mutex> lock(mutex_class_protection);

    stHailoStreamInfo* pStreamInfo = MultiNetworkPipeline::GetNetworkStreamInfoFromMatchingNetworkId(id_name);

    if (pStreamInfo) {

        if (get_from_output_stream) {

            for (size_t i = 0; i < pStreamInfo->NetVstreamOutputCount; i++) {
                NetworkVstreamInfo.push_back(pStreamInfo->NetVstreamOutputVstreamInfo[i]);
            }
        }
        else {

            for (size_t i = 0; i < pStreamInfo->NetVstreamInputCount; i++) {
                NetworkVstreamInfo.push_back(pStreamInfo->NetVstreamOutputVstreamInfo[i]);
            }

        }

        RetCode = MnpReturnCode::SUCCESS;
    }
    return RetCode;
}


MnpReturnCode MultiNetworkPipeline::Infer(const std::string &id_name, const std::vector<uint8_t> &data, std::string stream_id /*="default"*/, size_t input_stream_index /*=0*/)
{
    MnpReturnCode RetCode = MnpReturnCode::NOT_FOUND;       
    stHailoStreamInfo* pStreamInfo = nullptr;

    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;
    
    
    mutex_class_protection.lock();

    pStreamInfo = MultiNetworkPipeline::GetNetworkStreamInfoFromStreamChannel(id_name, stream_id);

    mutex_class_protection.unlock();

    if (pStreamInfo) {

        hailo_status status = HAILO_SUCCESS;
        status = hailo_vstream_write_raw_buffer(pStreamInfo->NetVstreamInputs[input_stream_index], 
                                                (void*)data.data(), 
                                                pStreamInfo->NetVstreamInputFrameSize[input_stream_index]);                                                
        REQUIRE_SUCCESS_CHECK(status, l_exit, "hailo_stream_sync_write_all_raw_buffer failed");
        RetCode = MnpReturnCode::SUCCESS;
    }

l_exit:
    
    return RetCode;
}



MnpReturnCode MultiNetworkPipeline::ReadOutputById(const std::string &id_name, std::vector<std::vector<float32_t>>& output_buffer, std::string stream_id /*= "default"*/)
{    
    MnpReturnCode RetCode = MnpReturnCode::NOT_FOUND;
    stHailoStreamInfo* pStreamInfo = nullptr;

    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;
    
    mutex_class_protection.lock();

    pStreamInfo = MultiNetworkPipeline::GetNetworkStreamInfoFromStreamChannel(id_name, stream_id);

    mutex_class_protection.unlock();

    if (pStreamInfo) {

       if (CheckOutputBuffer<float32_t>(pStreamInfo, output_buffer) != MnpReturnCode::SUCCESS) {
            RetCode = MnpReturnCode::INVALID_PARAMETER;
            DBG_ERROR("output_buffer needs to be initialized (InitializeOutputBuffer)");
            goto l_exit;
        }

        hailo_status status = HAILO_SUCCESS;
        for (size_t i = 0; i < pStreamInfo->NetVstreamOutputCount; i++) {

            status = hailo_vstream_read_raw_buffer( pStreamInfo->NetVstreamOutputs[i], 
                                                    output_buffer[i].data(), 
                                                    pStreamInfo->NetVstreamOutputFrameSize[i]);            

            REQUIRE_SUCCESS_CHECK(status, l_exit, "hailo_vstream_read_raw_buffer failed");                       

        }

        RetCode = MnpReturnCode::SUCCESS;
    }

l_exit:

    return RetCode;    
}


MnpReturnCode MultiNetworkPipeline::ReadOutputById(const std::string &id_name, std::vector<std::vector<uint8_t>>& output_buffer, std::string stream_id /*= "default"*/)
{

    MnpReturnCode RetCode = MnpReturnCode::NO_DATA_AVAILABLE;
    stHailoStreamInfo* pStreamInfo = nullptr;

    if (!hailo_device_found)
        return MnpReturnCode::HAILO_NOT_INITIALIZED;
    
    mutex_class_protection.lock();

    pStreamInfo = MultiNetworkPipeline::GetNetworkStreamInfoFromStreamChannel(id_name, stream_id);

    mutex_class_protection.unlock();

    if (pStreamInfo) {

        hailo_status status = HAILO_SUCCESS;

        if (CheckOutputBuffer<uint8_t>(pStreamInfo, output_buffer) != MnpReturnCode::SUCCESS) {
            RetCode = MnpReturnCode::INVALID_PARAMETER;
            DBG_ERROR("output_buffer needs to be initialized (InitializeOutputBuffer)");
            goto l_exit;
        }

        for (size_t i = 0; i < pStreamInfo->NetVstreamOutputCount; i++) {

            status = hailo_vstream_read_raw_buffer( pStreamInfo->NetVstreamOutputs[i], 
                                                    output_buffer[i].data(), 
                                                    pStreamInfo->NetVstreamOutputFrameSize[i]);

            REQUIRE_SUCCESS_CHECK(status, l_exit, "hailo_vstream_read_raw_buffer failed");                       
        }

        RetCode = MnpReturnCode::SUCCESS;
    }

l_exit:

    return RetCode;       

}
