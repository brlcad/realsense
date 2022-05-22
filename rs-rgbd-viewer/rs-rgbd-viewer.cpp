#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <librealsense2/rs.hpp>


static float get_depth_scale(rs2::device dev)
{
  for (rs2::sensor& sensor : dev.query_sensors()) {
    if (rs2::depth_sensor depth = rs2::depth_sensor(sensor)) {
      return depth.get_depth_scale();
    }
  }
  throw std::runtime_error("Device does not have a depth sensor");
}


/* Given a vector of streams, find a depth stream and (hopefully)
 * color stream to align depth with.
 */
static rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
  rs2_stream align_to = RS2_STREAM_ANY;
  bool depth_stream_found = false;
  bool color_stream_found = false;

  for (rs2::stream_profile sp : streams) {
    rs2_stream profile_stream = sp.stream_type();
    if (profile_stream != RS2_STREAM_DEPTH) {
      if (!color_stream_found) //Prefer color
        align_to = profile_stream;

      if (profile_stream == RS2_STREAM_COLOR) {
        color_stream_found = true;
      }
    } else {
      depth_stream_found = true;
    }
  }

  if (!depth_stream_found)
    throw std::runtime_error("No Depth stream available");

  if (align_to == RS2_STREAM_ANY)
    throw std::runtime_error("No stream found to align with Depth");

  return align_to;
}


static bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
  for(auto&& sp : prev) {
    //If previous profile is in current (maybe just added another)
    auto itr = std::find_if(std::begin(current), std::end(current),
                            [&sp](const rs2::stream_profile& current_sp) {
                              return sp.unique_id() == current_sp.unique_id();
                            });
    if (itr == std::end(current)) {
      return true;
    }
  }
  return false;
}


static void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
  const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
  uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

  int width = other_frame.get_width();
  int height = other_frame.get_height();
  int other_bpp = other_frame.get_bytes_per_pixel();

  for (int y = 0; y < height; y++) {
    auto depth_pixel_index = y * width;
    for (int x = 0; x < width; x++, ++depth_pixel_index) {
      // Get the depth value of the current pixel
      auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

      // Check if depth value is invalid (<=0) or over threshold
      if (pixels_distance <= 0.f || pixels_distance > clipping_dist) {
        // Calculate offset in other frame's buffer to current pixel
        auto offset = depth_pixel_index * other_bpp;

        // Set pixel to "background" color (0x999999)
        std::memset(&p_other_frame[offset], 0x01, other_bpp);
      }
    }
  }
}


/* this cannot be constructed repeatedly without crashing */
static rs2::colorizer c;

static int rs_read(rs2::pipeline_profile profile, rs2::pipeline p, cv::Mat& img, cv::Mat &dimg)
{
  /* overarching toggle for color or grayscale depth buffer image */
  bool show_color_depth = 0;
  rs2::frameset frames = p.wait_for_frames();

  rs2_stream align_to = find_stream_to_align(profile.get_streams());
  rs2::align align(align_to);
  auto processed = align.process(frames);

  rs2::frame color = frames.get_color_frame();
  const int w = color.as<rs2::video_frame>().get_width();
  const int h = color.as<rs2::video_frame>().get_height();
  cv::Mat nimg(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
  cv::cvtColor(nimg, img, cv::COLOR_BGR2RGB);

  rs2::video_frame other_frame = processed.first(align_to);
  rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

  rs2::frame depth;
  if (!aligned_depth_frame || !other_frame)
    depth = frames.get_depth_frame();

  float depth_clipping_distance = 10000.f;
  float depth_scale = get_depth_scale(profile.get_device());
  remove_background(other_frame, aligned_depth_frame, depth_scale, depth_clipping_distance);

  const int w_other = other_frame.get_width();
  const int h_other = other_frame.get_height();

  cv::Mat ndimg;
  if (show_color_depth) {
    rs2::video_frame depth_color=c.process(aligned_depth_frame);
    const int w_depth = depth_color.get_width();
    const int h_depth = depth_color.get_height();
    depth = depth_color;
    ndimg = cv::Mat(cv::Size(w_depth, h_depth), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
  } else {
    depth = aligned_depth_frame;
    const int dw = depth.as<rs2::video_frame>().get_width();
    const int dh = depth.as<rs2::video_frame>().get_height();
    ndimg = cv::Mat(cv::Size(dw, dh), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
    cv::resize(ndimg, dimg, cv::Size(w, h), cv::INTER_LINEAR);
    ndimg.convertTo(ndimg, CV_8UC1, 15 / 256.0);
  }

  //  cv::resize(ndimg, dimg, cv::Size(w, h), cv::INTER_LINEAR);
  //  ndimg.convertTo(ndimg, CV_8UC1, 15 / 256.0);
  dimg = ndimg;

  /* check if realsense config changed */
  if (profile_changed(p.get_active_profile().get_streams(), profile.get_streams())) {
    // if profile changed, update align object and get new depth scale
    profile = p.get_active_profile();
    align_to = find_stream_to_align(profile.get_streams());
    align = rs2::align(align_to);
    depth_scale = get_depth_scale(profile.get_device());
  }

  return 1;
}


int main(int ac, char *av[])
{
  int k = 0;

  const char *sourceRGB = "Source";
  const char *sourceDST = "Depth";

  int history = 100;
  double thresh = 400;
  bool shadows = true;
  cv::Ptr<cv::BackgroundSubtractor> KNN = cv::createBackgroundSubtractorKNN(history, thresh, shadows);
  cv::Ptr<cv::BackgroundSubtractor> MOG = cv::createBackgroundSubtractorMOG2(history, sqrt(thresh)/4.0, shadows);

  cv::Mat initial;
  cv::Mat dsrc = cv::Mat::zeros(initial.size(), CV_8UC1);
  rs2::pipeline pipe;

  rs2::config cfg;
  /* image dimensions */
  int w = 640;
  int n = 480;
  cfg.enable_stream(rs2_stream::RS2_STREAM_DEPTH, w, n, rs2_format::RS2_FORMAT_Z16);
  cfg.enable_stream(rs2_stream::RS2_STREAM_COLOR, w, n, rs2_format::RS2_FORMAT_RGB8);
  /* much slower processing
    cfg.enable_stream(rs2_stream::RS2_STREAM_DEPTH, 1280, 720, rs2_format::RS2_FORMAT_Z16);
    cfg.enable_stream(rs2_stream::RS2_STREAM_COLOR, 1280, 720, rs2_format::RS2_FORMAT_RGB8);
  */

  rs2::pipeline_profile profile = pipe.start(cfg);

  /* get units for depth pixels */
  rs2_stream align_to = find_stream_to_align(profile.get_streams());
  rs2::align align(align_to);

  /* actually read our realsense data */
  rs_read(profile, pipe, initial, dsrc);

  cv::Mat src = cv::Mat::zeros(initial.size(), CV_8UC3);

  initial = cv::Mat::zeros(initial.size(), CV_8UC3);

  /* initialize window positions */
  cv::imshow(sourceRGB, src);
  cv::moveWindow(sourceRGB, 0, 0);
  cv::imshow(sourceDST, dsrc);
  cv::moveWindow(sourceDST, 0, (initial.size().height+25)*2);

  while (k != 'q' && k != 27) {
    fflush(stdout);
    rs_read(profile, pipe, src, dsrc);
    // std::cout << "Width : " << src.size().width << std::endl;
    // std::cout << "Height: " << src.size().height << std::endl;

    /* do fun stuff */
    cv::Mat dMask;
    cv::inRange(dsrc, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), dMask);
    dsrc.setTo(cv::Scalar(255, 255, 255), dMask);
    // realsense reports a depth of 0
    bitwise_not(dsrc, dsrc);

    //    cv::Point minloc;
    //    cv::Point maxloc;
    double minval;
    double maxval;
    cv::minMaxLoc(dsrc, &minval, &maxval, NULL /*minloc*/, NULL /*&maxloc */);
    /* TODO: re-map intensities from 0 to 1 */

    /* display */
    cv::imshow(sourceRGB, src);
    cv::imshow(sourceDST, dsrc);

    k = cv::waitKey(200);
  }

  cv::destroyWindow(sourceRGB);
  cv::destroyWindow(sourceDST);

  return 0;
}
