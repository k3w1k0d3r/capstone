#ifndef NN_H
#define NN_H
#include "custom_position.h"
#include "tensorflow/cc/client/client_session.h"
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/array_ops.h>
#include "tensorflow/core/framework/tensor_slice.h"
#include <unistd.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <chrono>
#include <mutex>
#include <math.h>
#include <limits>

using namespace std;
using namespace tensorflow;
using namespace chrono;

namespace nn{
	class queue{
		int id = 0;
		vector<Tensor> inputs;
		vector<int> ids;
		unordered_map<int, vector<vector<float>>> results;
		unordered_map<int, bool> ready;
		std::mutex *run_mutex;
		std::mutex *ready_mutex;
		std::mutex *results_mutex;
		public:
			queue();
			int add_position(position c_position);
			vector<vector<float>> get_result(int id);
			void run(unique_ptr<Session> *session);
	};
	Status LoadGraph(const string& graph_file_name, unique_ptr<Session> *session);
	Tensor P2Tensor(position c_position);
	Tensor infer(Tensor input, unique_ptr<Session>& session);
}

#endif
