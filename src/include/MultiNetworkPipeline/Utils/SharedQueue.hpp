#ifndef _SharedQueue_H_
#define _SharedQueue_H_

#include <queue>
#include <mutex>
#include <condition_variable>

/* Declaration */

template <typename T>
class SharedQueue
{
public:
    SharedQueue();
    ~SharedQueue();
   
    
    T& front();
    void pop_front();

    void push_back(const T& item);
    void push_back(T&& item);

    int size();
    bool empty();
    void clear();

private:
    std::deque<T> queue_;
    std::mutex* mutex_;
    std::condition_variable cond_;
}; 
    
/* Implementation */

template <typename T>
SharedQueue<T>::SharedQueue(){
    mutex_ = new std::mutex();
}

template <typename T>
SharedQueue<T>::~SharedQueue(){
    delete mutex_;
}

template <typename T>
T& SharedQueue<T>::front()
{
    std::unique_lock<std::mutex> mlock(*mutex_);
    while (queue_.empty())
    {
        cond_.wait(mlock);
    }
    return queue_.front();
}

template <typename T>
void SharedQueue<T>::pop_front()
{
    std::unique_lock<std::mutex> mlock(*mutex_);
    while (queue_.empty())
    {
        cond_.wait(mlock);
    }
    queue_.pop_front();
}     

template <typename T>
void SharedQueue<T>::push_back(const T& item)
{    
    std::unique_lock<std::mutex> mlock(*mutex_);
    queue_.push_back(item);
    mlock.unlock();     // unlock before notificiation to minimize mutex con
    cond_.notify_one(); // notify one waiting thread

}

template <typename T>
void SharedQueue<T>::push_back(T&& item)
{    
    std::unique_lock<std::mutex> mlock(*mutex_);
    queue_.push_back(std::move(item));
    mlock.unlock();     // unlock before notificiation to minimize mutex con
    cond_.notify_one(); // notify one waiting thread

}

template <typename T>
int SharedQueue<T>::size()
{
    std::unique_lock<std::mutex> mlock(*mutex_);
    int size = queue_.size();
    mlock.unlock();
    return size;
}

template <typename T>
bool SharedQueue<T>::empty()
{
    std::unique_lock<std::mutex> mlock(*mutex_);
    bool isEmpty = queue_.empty();
    mlock.unlock();
    return isEmpty;
}

template <typename T>
void SharedQueue<T>::clear()
{
    std::unique_lock<std::mutex> mlock(*mutex_);
    
    std::deque<T> empty;
    std::swap( queue_, empty );
   
    mlock.unlock();
}

#endif //_SharedQueue_H_
