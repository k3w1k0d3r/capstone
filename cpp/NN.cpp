#include "NN.h"

using namespace std;
using namespace tensorflow;
using namespace ops;
using namespace chrono;

namespace nn{
	queue::queue(){
		this->run_mutex = new std::mutex();
		this->ready_mutex = new std::mutex();
		this->results_mutex = new std::mutex();
	}
	int queue::add_position(position c_position){
		vector<vector<int>> board = c_position.getboard();
		this->run_mutex->lock();
		Tensor input = P2Tensor(c_position);
		this->inputs.push_back(input);
		++this->id;
		this->ids.push_back(this->id);
		this->ready_mutex->lock();
		this->ready[this->id] = false;
		this->ready_mutex->unlock();
		int ret_val = this->id;
		this->run_mutex->unlock();
		return ret_val;
	}
	vector<vector<float>> queue::get_result(int id){
		bool ready = false;
		while(!ready){
			this->ready_mutex->lock();
			ready = this->ready[id];
			this->ready_mutex->unlock();
		}
		this->results_mutex->lock();
		vector<vector<float>> result = this->results[id];
		this->results.erase(id);
		this->results_mutex->unlock();
		this->ready_mutex->lock();
		this->ready.erase(id);
		this->ready_mutex->unlock();
		return result;
	}
	void queue::run(unique_ptr<Session> *session){
		if(this->inputs.size()==0){
			return;
		}
		this->run_mutex->lock();
		int count = min(4, (int)this->inputs.size());
		vector<Tensor> input = vector<Tensor>(this->inputs.begin(), this->inputs.begin()+count);
		Scope root = Scope::NewRootScope();
		ClientSession csession(root);
		Tensor tens_input = inputs[0];
		vector<Tensor> coutputs;
		for(int i = 1;i<count;++i){
			auto concat = Concat(root.WithOpName(to_string(i)), {tens_input, inputs[i]}, 0);
			csession.Run({concat}, &coutputs);
			tens_input = coutputs[0];
			coutputs.clear();
		}
		Tensor outputs = infer(tens_input, *session);
		vector<vector<float>> output;
		vector<float> temp_vec1;
		vector<float> temp_vec2;
		auto outputs_map = outputs.tensor<float, 2>();
		this->ready_mutex->lock();
		this->results_mutex->lock();
		for(int i = 0;i<count;++i){
			output.clear();
			temp_vec1.clear();
			temp_vec2.clear();
			for(int j = 0;j<362;++j){
				temp_vec1.push_back(outputs_map(i, j));
				temp_vec2.push_back(outputs_map(i+1, j));
			}
			output.push_back(temp_vec1);
			output.push_back(temp_vec2);
			this->results[this->ids[i]] = output;
			this->ready[this->ids[i]] = true;
		}
		this->results_mutex->unlock();
		this->ready_mutex->unlock();
		for(int i = 0;i<count;++i){
			this->inputs.erase(this->inputs.begin());
		}
		this->ids = vector<int>(this->ids.begin()+count, this->ids.end());
		this->run_mutex->unlock();
	}
	Tensor infer(Tensor input, unique_ptr<Session>& session){
		auto start = chrono::high_resolution_clock::now();
		vector<Tensor> outputs;
		Status run_status = session->Run({{"x", input}}, {"model/output_node/concat"}, {}, &outputs);
		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		//cout << duration.count() << endl;
		return outputs[0];
	}
	Status LoadGraph(const string& graph_file_name, unique_ptr<Session> *session){
		GraphDef graph_def;
		Status load_graph_status = ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);
		if(!load_graph_status.ok()){
			return load_graph_status;
		}
		auto options = SessionOptions();
		options.config.mutable_gpu_options()->set_allow_growth(true);
		session->reset(NewSession(options));
		Status session_create_status = (*session)->Create(graph_def);
		if(!session_create_status.ok()){
			return session_create_status;
		}
		return Status::OK();
	}
}
