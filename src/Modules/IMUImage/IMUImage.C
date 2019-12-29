// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Log.H>
#include <jevois/Core/ICM20948.H>
#include <jevois/Types/BoundedBuffer.H>
#include <list>
#include <chrono>
#include <unistd.h>
#include <mutex>
#include <cstdlib> // for std::system()
#include <fstream>
#include <opencv2/highgui/highgui.hpp>


// icon by Catalin Fertu in cinema at flaticon

//! JeVois sample module
/*! This module is provided as an example of how to create a new standalone module.

    JeVois provides helper scripts and files to assist you in programming new modules, following two basic formats:

    - if you wish to only create a single module that will execute a specific function, or a collection of such modules
      where there is no shared code between the modules (i.e., each module does things that do not relate to the other
      modules), use the skeleton provided by this sample module. Here, all the code for the sample module is compiled
      into a single shared object (.so) file that is loaded by the JeVois engine when the corresponding video output
      format is selected by the host computer.

    - if you are planning to write a collection of modules with some shared algorithms among several of the modules, it
      is better to first create machine vision Components that implement the algorithms that are shared among several of
      your modules. You would then compile all your components into a first shared library (.so) file, and then compile
      each module into its own shared object (.so) file that depends on and automatically loads your shared library file
      when it is selected by the host computer. The jevoisbase library and collection of components and modules is an
      example for how to achieve that, where libjevoisbase.so contains code for Saliency, ObjectRecognition, etc
      components that are used in several modules, and each module's .so file contains only the code specific to that
      module.

    @author Sample Author

    @videomapping YUYV 640 480 28.5 YUYV 640 480 28.5 zsk IMUImage
    @email sampleemail\@samplecompany.com
    @address 123 First Street, Los Angeles, CA 90012
    @copyright Copyright (C) 2017 by Sample Author
    @mainurl http://samplecompany.com
    @supporturl http://samplecompany.com/support
    @otherurl http://samplecompany.com/about
    @license GPL v3
    @distribution Unrestricted
    @restrictions None */
static jevois::ParameterCategory const ParamCateg("Save img and imu options");

JEVOIS_DECLARE_PARAMETER(bufsize, int, "Buf size for the queue",
                         2000, ParamCateg);
class IMUImage : public jevois::Module,
	public jevois::Parameter<bufsize>
{
  public:
    //! Default base class constructor ok
    using jevois::Module::Module;
    IMUImage(std::string const & instance) : jevois::Module(instance), 
	itsBufImgData(bufsize::get()), itsBufImgTime(bufsize::get()), itsStartSave(false), itsRunSaveIMU(true), itsRunSaveImg(true)
    {
       itsIMU = addSubComponent<jevois::ICM20948>("imu");
    }

	std::string time_str(){
	 time_t now = time(0);
	 //cout << "Number of sec since January 1,1970:" << now << endl;
	 tm *ltm = localtime(&now);

	 int year = 1900 + ltm->tm_year;
	 int month = 1 + ltm->tm_mon;
	 int day = ltm->tm_mday;
	 int hour = 1 + ltm->tm_hour;
	 int min = 1 + ltm->tm_min;
	 int sec = 1 + ltm->tm_sec;
	 char buf[256];
	 sprintf(buf, "%d_%02d_%02d_%02d_%02d_%02d", year, month, day, hour, min, sec);
	 return std::string(buf);
	}

    void postInit() override
    {
      //itsRunning.store(true);
      
      // Get our run() thread going, it is in charge of compresing and saving frames:
      //itsRunFut = std::async(std::launch::async, &SaveVideo::run, this);
	  LINFO("run async save imu");
	  itsRunSaveIMU.store(true);
	  itsStartSave.store(false);
      futureSaveIMU = std::async(std::launch::async, &IMUImage::saveIMU, this);

	  LINFO("run async save img");
	  itsRunSaveImg.store(true);
      futureSaveImg = std::async(std::launch::async, &IMUImage::saveImg, this);

    }

	void postUninit() override
    {
	  // stop save imu thread
	  itsRunSaveIMU.store(false);
	  itsStartSave.store(false);
      try { futureSaveIMU.get(); } catch (...) { jevois::warnAndIgnoreException(); }

	  // stop save image thread
	  itsRunSaveImg.store(false);
	  itsBufImgData.push(cv::Mat());
	  itsBufImgTime.push(-1);
      try { futureSaveImg.get(); } catch (...) { jevois::warnAndIgnoreException(); }
    }

    //! Virtual destructor for safe inheritance
    virtual ~IMUImage() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {

      cv::Mat imgray = inframe.getCvGRAY();
      outframe.sendCv(imgray);
	  if (itsStartSave.load()){
		  //todo: save image to a directory
		  auto now = std::chrono::high_resolution_clock::now();
		  auto micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
		  itsBufImgData.push(imgray);
		  itsBufImgTime.push(micro);
		  //std::string imPNGpath = imgPath+"/"+std::to_string(micro)+".png";
		  //cv::imwrite(imPNGpath, imgray);
		  if (nimgs++%100==0){
			  LINFO("save image: " << nimgs);
		  }
	  }
    }

	void process(jevois::InputFrame && inframe) override
    {
      cv::Mat imgray = inframe.getCvGRAY();
	  if (itsStartSave.load()){
		  //todo: save image to a directory
		  auto now = std::chrono::high_resolution_clock::now();
		  auto micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
		  itsBufImgData.push(imgray);
		  itsBufImgTime.push(micro);
		  //std::string imPNGpath = imgPath+"/"+std::to_string(micro)+".png";
		  //cv::imwrite(imPNGpath, imgray);
		  if (nimgs++%100==0){
			  LINFO("save image: " << nimgs);
		  }
	  }
    }

	void parseSerial(std::string const & str, std::shared_ptr<jevois::UserInterface> s) override
     {
       if (str == "start")
       {
		 nimgs = 0;
         itsStartSave.store(true);
		 std::string timepath = time_str();
		 imuImgRootPath = jevois_root_path + "/"+timepath;
		 imgPath = imuImgRootPath + "/left/" ;
		 std::string cmd = "/bin/mkdir -p " + imgPath;
		 LINFO("cmd: " << cmd);
		 std::system(cmd.c_str());
         sendSerial("SAVESTART");
       }
       else if (str == "stop")
       {
         itsStartSave.store(false);
		 itsBufImgData.push(cv::Mat());
		 itsBufImgTime.push(-1);
		 while (itsBufImgData.filled_size())
		 {
			 LINFO("Waiting for writer thread to complete, " << itsBufImgData.filled_size() << " frames to go...");
			 std::this_thread::sleep_for(std::chrono::milliseconds(200));
		 }
         sendSerial("SAVESTOP");
       }
       else throw std::runtime_error("Unsupported module command");
     }

	 void supportedCommands(std::ostream & os) override
     {
       os << "start - start saving video" << std::endl;
       os << "stop - stop saving video and increment video file number" << std::endl;
     }


  protected:
	void saveImg(){
		while(itsRunSaveImg.load()){
			while (itsBufImgData.filled_size()){
				if (itsBufImgData.filled_size() != itsBufImgTime.filled_size()){
					//std::runtime_error
					LFATAL("time data size != img data size");
				}
				cv::Mat im = itsBufImgData.pop();
				long long time = itsBufImgTime.pop();
				if (im.empty() && time < 0){
					break;
				}
				std::string imPNGpath = imgPath+"/"+std::to_string(time)+".png";
				cv::imwrite(imPNGpath, im);
			}
			usleep(30*1e3);
		}
	}

    void saveIMU(){
	  while(itsRunSaveIMU.load()){
		  if (itsStartSave.load()){
			  unsigned int n = 0;
			  std::string imuFile = imuImgRootPath+"/imu0.csv";
			  LINFO("imu file path: " << imuFile);
			  std::ofstream fimu(imuFile.c_str(), std::ofstream::out);
			  while (itsStartSave.load()){
				  jevois::IMUdata d = itsIMU->get();
				//  if (itsIMU->dataReady() > 10)
				//	  continue;
				  auto now = std::chrono::high_resolution_clock::now();
				  auto micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
				  fimu << micro << "," << d.ax()*g0 << "," << d.ay()*g0 << "," << d.az()*g0 << 
					  "," << d.gx()*deg2rad << "," << d.gy()*deg2rad << "," << d.gz()*deg2rad 
					  << "," << d.mx() << "," << d.my() << "," << d.mz() << std::endl;
				  if (n++%100 == 0){
					  //LINFO("imu get");
					  LINFO("a: " << d.ax() << " " << d.ay() << " " << d.az() << " g: " << d.gx() << " " << d.gy() << " " << d.gz() << " mag: " << d.mx() << " " << d.my() << " " << d.mz());
					  LINFO("mode: " << itsIMU->mode::get() << " grate: " << itsIMU->grate::get() << " data ready: " << itsIMU->dataReady() << " mill: " << micro);
				  }
				  usleep(9*1e3);
			  }
			  fimu.close();
			  LINFO("imu file closed");
		  }
		  else
			  usleep(100*1e3);
	  }
    }

	//bool checkStartSaveIMU(){
	//	  std::unique_lock<std::mutex> lck (startIMUMtx);
	//	  return itsStartIMU;
	//}
  private:
	jevois::BoundedBuffer<cv::Mat, jevois::BlockingBehavior::Block, jevois::BlockingBehavior::Block> itsBufImgData;
	jevois::BoundedBuffer<long long, jevois::BlockingBehavior::Block, jevois::BlockingBehavior::Block> itsBufImgTime;
    std::shared_ptr<jevois::ICM20948> itsIMU;
    std::list<jevois::IMUdata> itsIMUdata;
    std::atomic<bool> itsRunSaveIMU, itsStartSave, itsRunSaveImg;
	std::future<void> futureSaveIMU, futureSaveImg;
	const std::string jevois_root_path = "/jevois/data/saveIMUImage";
	std::string imgPath, imuImgRootPath;
	const float g0 = 9.8, deg2rad = 3.1415926/180.;
	int nimgs;
	//bool itsStartIMU;
	//std::mutex startIMUMtx;
	//std::condition_variable startIMUCV;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(IMUImage);
