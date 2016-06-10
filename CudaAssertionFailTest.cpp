
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

#include <thread>

using namespace std;

//-----------------------------------------------------------------------------
// PASS
//-----------------------------------------------------------------------------
void RunAdd()
{
  cv::cuda::GpuMat Row1(cv::Mat::ones(1, 100, CV_8U));
  cv::cuda::GpuMat Row2(cv::Mat::ones(1, 100, CV_8U));
  cv::cuda::GpuMat Row3(cv::Mat::ones(1, 100, CV_8U));
  cv::cuda::GpuMat Row4(cv::Mat::ones(1, 100, CV_8U));

  cv::cuda::add(Row1, Row2, Row3, Row4);
}

//-----------------------------------------------------------------------------
// FAIL
//-----------------------------------------------------------------------------
void RunMinMax()
{
  double MaxVal;

  cv::cuda::minMax(
    cv::cuda::GpuMat(cv::Mat::ones(100, 1, CV_32S)),
    0,
    &MaxVal);
}

//-----------------------------------------------------------------------------
// FAIL
//-----------------------------------------------------------------------------
void RunSqrSum()
{
  cv::cuda::GpuMat Row1(cv::Mat::ones(1, 100, CV_8U));
  cv::cuda::GpuMat Row2(cv::Mat::ones(1, 100, CV_8U));

  cv::cuda::sqrSum(Row1, Row2);
}

//-----------------------------------------------------------------------------
// FAIL
//-----------------------------------------------------------------------------
void RunSum()
{
  cv::cuda::GpuMat dIsMaximalMask(cv::Mat::ones(100, 1, CV_32S));

  cv::cuda::sum(dIsMaximalMask);
}

//-----------------------------------------------------------------------------
// PASS
//-----------------------------------------------------------------------------
void RunSurf()
{
  cv::cuda::GpuMat Image(cv::Mat::zeros(500, 500, CV_8UC1));
  vector<cv::KeyPoint> CurMetaFeatures;
  cv::cuda::SURF_CUDA Surf;

  Surf(Image, cv::cuda::GpuMat(), CurMetaFeatures, Image);
}

//*****************************************************************************
//*****************************************************************************
int main(int argc, const char** argv)
{
  thread Thread1(RunSum);
  thread Thread2(RunMinMax);

  Thread1.join();
  Thread2.join();

  return 0;
}
