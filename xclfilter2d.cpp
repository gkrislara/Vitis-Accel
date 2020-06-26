#include<xcl2.hpp>
#include<CL/cl2.hpp>
#include<cl_ext_xilinx.h>
#include "common/xf_headers.hpp"

#define FILTER2D_KERNEL_NAME "filter2d_accel"

#define fourcc	0x56595559

short int usmcoeff[9]={1,-2,1,-2,6,-2,1,-2,1};

int main(int argc,char** argv)
{
	cv::Mat in_rgb,out_rgb,in_ycrcb,out_ycrcb;
    int rows,cols;

	//read RGB image and convert to YCrCb
	in_rgb= cv::imread(argv[2],1);
	if (!in_rgb.data) {
	        std::cout << "ERROR: Cannot open image " << argv[2] << std::endl;
	        return EXIT_FAILURE;
	}
	cv::cvtColor(in_rgb, in_ycrcb,CV_RGB2YCrCb);
    rows=in_rgb.rows;
    cols=in_rgb.cols;

	//create host memory for output images
	//out_rgb.create(in_rgb.rows, in_rgb.cols, in_rgb.depth());
	out_ycrcb.create(in_ycrcb.rows, in_ycrcb.cols, in_ycrcb.depth());
//------------------------------OPENCL----------------------
	size_t image_in_size_bytes = in_ycrcb.rows * in_ycrcb.cols * in_ycrcb.depth()* sizeof(unsigned char);
	size_t image_out_size_bytes = image_in_size_bytes;

	std::cout<<"OCL Begin \n";
    cl_int err;
	//get device and device name
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];
	std::string device_name = device.getInfo<CL_DEVICE_NAME>();

	//create context
	OCL_CHECK(err,cl::Context context(device, NULL, NULL, NULL,&err));
	if(err!=CL_SUCCESS)
	{
		std::cout<<"Context not created\n";
	}
	else std::cout<<"Context created\n";


	//create command queue
	OCL_CHECK(err,cl::CommandQueue q(context, device, 0,&err));
	if(err!=CL_SUCCESS)
		{
			std::cout<<"queue not created\n";
		}
	else std::cout<<"queue created\n";

	//load binary -> cl::program()
	auto fileBuf = xcl::read_binary_file(argv[1]);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	devices.resize(1);
	OCL_CHECK(err,cl::Program Program(context,devices, bins,NULL,&err));
	if(err!=CL_SUCCESS)
		{
			std::cout<<"unable to program device\n";
		}
	else std::cout<<"device programmed with xclbin\n";

	//create kernel
    OCL_CHECK(err,cl::Kernel kernel(Program,FILTER2D_KERNEL_NAME,&err));
    if(err!=CL_SUCCESS)
    	{
    		std::cout<<"Kernel not created \n ";
    	}
    else std::cout<<"Kernel created \n";

    //Allocate Buffers - create memory in global memory
    OCL_CHECK(err,cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes,&err));
    if(err!=CL_SUCCESS)
    	{
    		std::cout<<"Input image buffer (YCrCb) not created \n";
    	}
    else std::cout<<"Input image buffer (YCrCb) created \n";

    OCL_CHECK(err,cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes,&err));
    if(err!=CL_SUCCESS)
       	{
       		std::cout<<"Output image buffer (YCrCb) not created \n";
       	}
    else std::cout<<"Output image buffer (YCrCb) created \n";
    size_t filter_size=3*3*2*sizeof(unsigned char);
    cl::Buffer usm(context, CL_MEM_READ_ONLY,filter_size);

    //set Kernel arguments
    OCL_CHECK(err,err = kernel.setArg(0,buffer_inImage)); //maxi
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Argument 0 not set \n";
           	}
    else std::cout<<"Kernel Argument 0 set \n";
    OCL_CHECK(err,err = kernel.setArg(1,buffer_outImage)); //maxi
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Argument 1 not set \n";
           	}
    else std::cout<<"Kernel Argument 1 set \n";
    OCL_CHECK(err,err = kernel.setArg(2,usm)); //saxi
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Argument 2 not set \n";
           	}
    else std::cout<<"Kernel Argument 2 set \n";
    OCL_CHECK(err,err = kernel.setArg(3,rows)); //saxi
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Argument 3 not set \n";
           	}
    else std::cout<<"Kernel Argument 3 set \n";
    OCL_CHECK(err,err = kernel.setArg(4,cols)); //saxi
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Argument 4 not set \n";
           	}
    else std::cout<<"Kernel Argument 4 set \n";
    OCL_CHECK(err,err = kernel.setArg(5,fourcc));
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Argument 5 not set \n";
           	}
    else std::cout<<"Kernel Argument 5 set \n";
    OCL_CHECK(err,err = kernel.setArg(6,fourcc));
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Argument 6 not set \n";
           	}
    else std::cout<<"Kernel Argument 6 set \n";

    //create event
    cl::Event event_usm;

    //write data to kernel -> enqueue write buffer
    OCL_CHECK(err,err=q.enqueueWriteBuffer(buffer_inImage,CL_TRUE,0,image_in_size_bytes,in_ycrcb.data,nullptr,&event_usm));
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Write Image Buffer fail \n";
           	}
    else std::cout<<"Writing into Image Buffer ... \n";

    OCL_CHECK(err,err=q.enqueueWriteBuffer(usm,CL_TRUE,0,filter_size,usmcoeff,nullptr,&event_usm));
    if(err!=CL_SUCCESS)
            {
          		std::cout<<"Write Filter Buffer fail \n";
            }
    else std::cout<<"Writing into Filter Buffer ... \n";

    //enqueue task -> execute kernel
    OCL_CHECK(err,err=q.enqueueTask(kernel));
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Kernel Activation failed \n";
           	}
    else std::cout<<"Kernel Activated!\n";

    //read the data from kernel -> enqueue read buffer
    OCL_CHECK(err,err =q.enqueueReadBuffer(buffer_outImage,CL_TRUE,0,image_out_size_bytes,out_ycrcb.data,nullptr,&event_usm));
    if(err!=CL_SUCCESS)
           	{
           		std::cout<<"Read Buffer fail \n";
           	}
    else std::cout<<" Reading Buffer...... \n";


    //clean up -> release buffers
    q.finish();
    std::cout<<"OCL COMPLETE!\n";
//----------------------------------------------------------------------------------------------------
    //write RGB image
    cv::cvtColor(out_ycrcb, out_rgb,CV_YCrCb2RGB);
	cv::imwrite("output.png",out_rgb);



}
