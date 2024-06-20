#ifndef _Timer_H_
#define _Timer_H_

#include <chrono>

class Timer
{
private:
  using clock_type = std::chrono::high_resolution_clock;
  using second_type = std::chrono::duration<double, std::chrono::seconds::period>;

  std::chrono::time_point<clock_type> m_startTime;

public:
  Timer() : m_startTime(clock_type::now())
  {
  }

  double reset()
  {
    double dt = getElapsedInSec();
    m_startTime = clock_type::now();
    return dt;
  }

  double getElapsedInSec() const
  {
    return std::chrono::duration_cast<second_type>(clock_type::now() - m_startTime).count();
  }
  
  bool isTimePastSec(double seconds) const
  {
      return ((getElapsedInSec() - seconds) > 0.0f) ? true : false;
  }
};


class TimerMs
{
private:
  using clock_type = std::chrono::high_resolution_clock;
  using second_type = std::chrono::duration<double, std::chrono::milliseconds::period>;

  std::chrono::time_point<clock_type> m_startTime;

public:
  TimerMs() : m_startTime(clock_type::now())
  {
  }

  double reset()
  {
    double dt = getElapsedInMs();
    m_startTime = clock_type::now();
    return dt;
  }

  size_t getStartTime() const
  {
    return std::chrono::duration_cast<second_type>(m_startTime.time_since_epoch()).count();
  }

  double getElapsedInMs() const
  {
    return std::chrono::duration_cast<second_type>(clock_type::now() - m_startTime).count();
  }
  
  bool isTimePastMs(double milliseconds) const
  {
      return ((getElapsedInMs() - milliseconds) > 0.0f) ? true : false;
  }
};


#endif //_Timer_H_