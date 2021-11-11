#ifndef SEARCH_H
#define SEARCH_H
#include "treestate.h"
#include "game.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include <stdlib.h>
#include <limits>
#include <thread>

using namespace std;
using namespace chrono;

namespace search{
	void simulate(treestate* init_state, int total);
	void queue_run(nn::queue *pos_queue, unique_ptr<Session> *session, atomic_bool *queue_needed);
	tuple<int, treestate*, vector<double>> choose(float temperature, game& p_game, position c_position, int turn, unique_ptr<Session> *session, treestate* init_state, nn::queue *pos_queue, bool use_queue, int iterations, int thread_count, float epsilon, float Alpha);
	tuple<int, vector<int>, vector<double>, vector<long>, vector<long>> playgame(game& p_game, unique_ptr<Session> *session, float& temperature, bool& use_queue, int iterations, int thread_count, float epsilon, float Alpha);
	bool testgame(game& p_game, unique_ptr<Session> *session1, unique_ptr<Session> *session2,float& temperature, bool& use_queue, int iterations, int thread_count, float epsilon, float Alpha, bool tested);
}

#endif
