#ifndef TREESTATE_H
#define TREESTATE_H
#include <limits>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <math.h>
#include "game.h"
#include "NN.h"

using namespace std;

class treestate{
	position c_position;
	int visits;
	int v_loss;
	bool keep = false;
	float P;
	vector<float> Q;
	vector<float> W;
	vector<float> Q_improved;
	int turn;
	int player;
	game s_game;
	treestate *parent;
	float c;
	bool use_queue;
	vector<treestate *> children;
	int move;
	nn::queue *pos_queue;
	unique_ptr<Session> *session;
	float score;
	bool score_good = false;
	std::mutex *finish_mutex;
	std::mutex *back_mutex;
	std::mutex *infer_mutex;
	std::mutex *choose_mutex;
	public:
		treestate(treestate *parent, int move, float P);
		treestate(float c, game& s_game, position c_position, int turn, nn::queue *pos_queue, unique_ptr<Session> *session, bool use_queue);
		vector<float> finish_sim();
		bool is_leaf();
		treestate *backprop(vector<float> Q);
		float get_score();
		treestate *choose_child();
		position getposition();
		int getvisits();
		int getmove();
		vector<tuple<int, int, treestate*>> getresults();
		void mark_keep();
		void del_inf_mut();
		~treestate();
};

#endif
