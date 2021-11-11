#include "treestate.h"
#include "game.h"
#include "search.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include <stdlib.h>
#include <limits>
#include <thread>

using namespace std;
using namespace chrono;

namespace search{
	void simulate(treestate* init_state, int total){
		bool leaf;
		bool finished;
		int new_move;
		int winner;
		treestate *c_state = init_state;
		vector<float> Q;
		for(int i = 0;i<total;++i){
			leaf = c_state->is_leaf();
			while(!leaf){
				c_state = c_state->choose_child();
				leaf = c_state->is_leaf();
			}
			Q = c_state->finish_sim();
			c_state = c_state->backprop(Q);
		}
	}
	void queue_run(nn::queue *pos_queue, unique_ptr<Session> *session, atomic_bool *queue_needed){
		while(*queue_needed){
			pos_queue->run(session);
		}
	}
	tuple<int, treestate*, vector<double>> choose(float temperature, game& p_game, position c_position, int turn, unique_ptr<Session> *session, treestate* init_state, nn::queue *pos_queue, bool use_queue, int iterations, int thread_count, float epsilon, float Alpha){
		treestate *c_state = init_state;
		if(!c_state){
			c_state = new treestate(sqrt(2), p_game, c_position, turn, pos_queue, session, use_queue);
		}
		int total = (iterations-c_state->getvisits())/thread_count;
		int remainder = (iterations-c_state->getvisits())%thread_count;
		atomic_bool queue_needed(use_queue);
		std::thread queue_thread(queue_run, pos_queue, session, &queue_needed);
		vector<std::thread> threads;
		for(int i = 0;i<thread_count;++i){
			threads.push_back(std::thread(simulate, c_state, total+remainder*(int)(i==(thread_count-1))));
		}
		for(int i = 0;i<threads.size();++i){
			threads[i].join();
		}
		queue_needed = false;
		queue_thread.join();
		vector<tuple<int, int, treestate*>> results = c_state->getresults();
		gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
		gsl_rng_set(r, time(NULL));
		double theta[results.size()];
		double alpha[results.size()];
		for(int i = 0;i<results.size();++i){
			alpha[i] = Alpha;
		}
		gsl_ran_dirichlet(r, results.size(), alpha, theta);
		int move;
		treestate* next_state;
		double P[results.size()];
		vector<double> P_vec;
		int j = 0;
		vector<int> moveset = p_game.getmoveset();
		for(int i = 0;i<moveset.size();++i){
			P_vec.push_back(0);
		}
		for(int i = 0;i<results.size();++i){
			P[i] = (1-epsilon)*pow(get<0>(results[i]), 1/temperature)+epsilon*theta[i];
			P_vec[get<1>(results[i])] = P[i];
		}
		gsl_ran_discrete_t *g = gsl_ran_discrete_preproc(results.size(), P);
		int choice = gsl_ran_discrete(r, g);
		gsl_rng_free(r);
		gsl_ran_discrete_free(g);
		next_state = get<2>(results[choice]);
		next_state->mark_keep();
		move = get<1>(results[choice]);
		delete c_state;
		return make_tuple(move, next_state, P_vec);
	}
	tuple<int, vector<int>, vector<double>, vector<long>, vector<long>> playgame(game& p_game, unique_ptr<Session> *session, float& temperature, bool& use_queue, int iterations, int thread_count, float epsilon, float Alpha){
		nn::queue *pos_queue = new nn::queue();
		position c_position = p_game.getposition();
		int turn = 0;
		int move;
		treestate *c_state = NULL;
		tuple<int, treestate*, vector<double>> choice;
		bool finished = get<0>(p_game.outcome(c_position));
		vector<int> boardlist;
		vector<int> tempdata;
		vector<double> temppolicy;
		vector<double> policylist;
		while(!finished){
			choice = choose(temperature, p_game, c_position, turn, session, c_state, pos_queue, use_queue, iterations, thread_count, epsilon, Alpha);
			c_state = get<1>(choice);
			c_position = c_state->getposition();
			move = get<0>(choice);
			finished = get<0>(p_game.outcome(c_position));
			tempdata = c_position.getboardflat();
			temppolicy = get<2>(choice);
			boardlist.insert(boardlist.end(), tempdata.begin(), tempdata.end());
			policylist.insert(policylist.end(), temppolicy.begin(), temppolicy.end());
			++turn;
		}
		c_state->del_inf_mut();
		delete c_state;
		delete pos_queue;
		vector<long> dims = c_position.getdims();
		dims[0] = turn;
		vector<long> p_dims = {turn, (long)temppolicy.size()};
		return make_tuple(get<1>(p_game.outcome(c_position)), boardlist, policylist, dims, p_dims);
	}
	bool testgame(game& p_game, unique_ptr<Session> *session1, unique_ptr<Session> *session2, float& temperature, bool& use_queue, int iterations, int thread_count, float epsilon, float Alpha, bool tested){
		nn::queue *pos_queue = new nn::queue();
		position c_position = p_game.getposition();
		int turn = 0;
		int move;
		vector<treestate*> c_state = {NULL, NULL};
		vector<unique_ptr<Session>*> session = {session1, session2};
		tuple<int, treestate*, vector<double>> choice;
		bool finished = get<0>(p_game.outcome(c_position));
		treestate* next_state;
		while(!finished){
			choice = choose(temperature, p_game, c_position, turn, session[turn%2], c_state[turn%2], pos_queue, use_queue, iterations, thread_count, epsilon, Alpha);
			c_state[turn%2] = get<1>(choice);
			if(c_state[1-(turn%2)]){
				next_state = c_state[1-(turn%2)]->getstate(get<0>(choice));
				if(next_state){
					next_state->mark_keep();
				}
				delete c_state[1-(turn%2)];
				c_state[1-(turn%2)] = next_state;
			}
			c_position = c_state[turn%2]->getposition();
			move = get<0>(choice);
			finished = get<0>(p_game.outcome(c_position));
			++turn;
		}
		c_state[0]->del_inf_mut();
		delete c_state[0];
		c_state[1]->del_inf_mut();
		delete c_state[1];
		delete pos_queue;
		return 2*tested-1==get<1>(p_game.outcome(c_position));
	}
}
