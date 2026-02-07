#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>
#include <memory>
#include <type_traits>

class ThreadPool {
private:
    struct Task {
        virtual ~Task() = default;
        virtual void execute() = 0;
    };

    struct StdFunctionTask : Task {
        std::function<void()> func;
        StdFunctionTask(std::function<void()> f) : func(std::move(f)) {}
        void execute() override { func(); }
    };
    std::vector<std::thread> workers;

    // task queue
    std::queue<std::shared_ptr<Task>> tasks;

    // lock
    mutable std::mutex queue_mutex;
    std::mutex complete_mutex;
    std::condition_variable condition;
    std::condition_variable complete_cv;

    // status flags
    std::atomic<bool> stop;
    std::atomic<int> working;
    std::atomic<int> task_num;

public:
    //std::thread::hardware_concurrency()
    ThreadPool(int threadCount) : stop(false), working(0), task_num(0)
    {
        if (threadCount == 0) 
            threadCount = 1;

        workers.reserve(threadCount);
        for (int i = 0; i < threadCount; ++i) {
            workers.emplace_back([this, i]() {
                workerLoop(i);
                });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();

        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // submit task
    void enqueue(std::function<void()> job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            tasks.emplace(std::make_shared<StdFunctionTask>(std::move(job)));
            ++task_num;
        }
        condition.notify_one();
    }

    // wait for all tasks complete
    void waitAll() {
        std::unique_lock<std::mutex> lock(complete_mutex);
        complete_cv.wait(lock, [this]() {
            return tasks.empty() && working == 0 && task_num == 0;
            });
    }

    int getThreadCount() const {
        return workers.size();
    }

    // get the number of tasks in the queue
    int getQueueSize() const {
        std::unique_lock<std::mutex> lock(queue_mutex);
        return tasks.size();
    }

private:

    void workerLoop(int thread_id) {
        while (true) {
            std::shared_ptr<Task> task;

            // get task
            {
                std::unique_lock<std::mutex> lock(queue_mutex);

                // wait for task
                condition.wait(lock, [this]() {
                    return stop || (!tasks.empty());
                    });

                if (stop && tasks.empty()) {
                    return;
                }


                // get task
                task = std::move(tasks.front());
                tasks.pop();
                working++;
            }

            // do task
            task->execute();
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                working--;
                task_num--;

                // notify all
                if (tasks.empty() && working == 0) {
                    std::lock_guard<std::mutex> lock(complete_mutex);
                    complete_cv.notify_all();
                }
            }
        }
    }
};
