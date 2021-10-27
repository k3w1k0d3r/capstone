#include "treestate.h"

using namespace std;
using namespace tensorflow;

treestate::treestate(treestate *parent, int move, float P){
	this->visits = 0;
	this->v_loss = 0;
	this->parent = parent;
	this->infer_mutex = this->parent->infer_mutex;
	this->use_queue = this->parent->use_queue;
	this->finish_mutex = new std::mutex();
	this->back_mutex = new std::mutex();
	this->choose_mutex = new std::mutex();
	this->s_game = this->parent->s_game;
	vector<int> player_order = this->s_game.getplayerorder();
	this->P = P;
	for(int i = 0;i<player_order.size();++i){
		this->Q.push_back(0);
		this->W.push_back(0);
		this->Q_improved.push_back(0);
	}
	this->turn = this->parent->turn+1;
	this->player = this->s_game.getplayer(this->turn);
	this->move = move;
	this->c_position = this->s_game.getposition(this->move, this->parent->c_position, this->parent->player);
	this->c = this->parent->c;
	this->pos_queue = this->parent->pos_queue;
	this->session = this->parent->session;
	this->score = -numeric_limits<float>::infinity();
}
treestate::treestate(float c, game& s_game, position c_position, int turn, nn::queue *pos_queue, unique_ptr<Session> *session, bool use_queue){
	this->infer_mutex = new std::mutex();
	this->back_mutex = new std::mutex();
	this->finish_mutex = new std::mutex();
	this->choose_mutex = new std::mutex();
	this->visits = 0;
	this->v_loss = 0;
	this->use_queue = use_queue;
	this->parent = NULL;
	this->score = -numeric_limits<float>::infinity();
	this->s_game = s_game;
	this->turn = turn;
	this->player = this->s_game.getplayer(this->turn);
	this->c_position = c_position;
	this->c = c;
	this->pos_queue = pos_queue;
	this->session = session;
	vector<int> player_order = this->s_game.getplayerorder();
	for(int i = 0;i<player_order.size();++i){
		this->Q.push_back(0);
		this->W.push_back(0);
		this->Q_improved.push_back(0);
	}
}
vector<float> treestate::finish_sim(){
	vector<int> player_order = this->s_game.getplayerorder();
	this->infer_mutex->lock();
	Tensor NN_out_tensor = nn::infer(nn::P2Tensor(this->c_position), *(this->session));
	this->infer_mutex->unlock();
	auto NN_out = NN_out_tensor.tensor<float, 2>(); //felt lazy, stopped making the queue work. Queue didn't seem to do much for speed anyway
	float logit;
	float sum = 0;
	vector<float> policy;
	tuple<bool, int> outcome = this->s_game.outcome(this->c_position);
	bool finished = get<0>(outcome);
	int winner = get<1>(outcome);
	for(int i = 0;i<player_order.size();++i){
		this->Q[i] = NN_out(i, this->s_game.getmoveset().size());
		if(finished){
			this->Q[i] = 2*(player_order[i]==winner)-1;
		}
		else if(player_order[i]==this->player){
			for(int j = 0;j<this->s_game.getmoveset().size();++j){
				logit = NN_out(i, s_game.getmoveset()[j]);
				if(!this->s_game.movelegal(this->c_position, s_game.getmoveset()[j])){
					logit = numeric_limits<float>::infinity();
				}
				policy.push_back(exp(logit));
				sum+=policy[policy.size()-1];
			}
			for(int j = 0;j<this->s_game.getmoveset().size();++j){
				if(this->s_game.movelegal(this->c_position, s_game.getmoveset()[j])){
					treestate *new_child = new treestate(this, s_game.getmoveset()[j], policy[j]/sum);
					new_child->score = new_child->get_score();
					this->children.push_back(new_child);
				}
			}
		}
	}
	this->score_good = true;
	this->finish_mutex->unlock();
	return this->Q;
}
bool treestate::is_leaf(){
	bool finished = get<0>(this->s_game.outcome(this->c_position));
	this->finish_mutex->lock();
	bool leaf = finished||this->children.size()==0;
	if(!leaf){
		this->finish_mutex->unlock();
	}
	return leaf;
}
treestate *treestate::backprop(vector<float> Q){
	this->back_mutex->lock();
	if(this->parent){
		this->parent->choose_mutex->lock();
		this->parent->score_good = false;
	}
	++this->visits;
	--this->v_loss;
	for(int i = 0;i<Q.size();++i){
		this->W[i]+=Q[i];
		this->Q_improved[i] = W[i]/this->visits;
	}
	this->back_mutex->unlock();
	if(this->parent){
		this->parent->choose_mutex->unlock();
		return this->parent->backprop(Q);
	}
	return this;
}
float treestate::get_score(){
	if(this->parent){
		for(int i = 0;i<this->s_game.getplayerorder().size();++i){
			if(this->s_game.getplayerorder()[i]==this->parent->player){
				return this->Q_improved[i]+this->c*this->P*sqrt(this->parent->visits+this->parent->v_loss)/(1+this->visits+this->v_loss);
			}
		}
	}
	return 0;
}
treestate *treestate::choose_child(){
	this->back_mutex->lock();
	this->choose_mutex->lock();
	treestate *best_child = NULL;
	float best_score = -numeric_limits<float>::infinity();
	if(this->score_good){
		for(int i = 0;i<this->children.size();++i){
			if(this->children[i]->score>best_score){
				best_child = this->children[i];
				best_score = this->children[i]->score;
			}
		}
	}
	else{
		for(int i = 0;i<this->children.size();++i){
			this->children[i]->score = this->children[i]->get_score();
			if(this->children[i]->score>best_score){
				best_child = this->children[i];
				best_score = this->children[i]->score;
			}
		}
	}
	++best_child->v_loss;
	if(!this->parent){
		++this->v_loss;
	}
	this->score_good = true;
	this->choose_mutex->unlock();
	this->back_mutex->unlock(); //backprop can probably get a bit out of order here but I think v_loss makes it work and I don't want to deal with it.
	return best_child;
}
position treestate::getposition(){
	return this->c_position;
}
int treestate::getvisits(){
	float visits = this->visits;
	return visits;
}
int treestate::getmove(){
	return this->move;
}
vector<tuple<int, int, treestate*>> treestate::getresults(){
	vector<tuple<int, int, treestate*>> out;
	for(int i = 0;i<this->children.size();++i){
		out.push_back(make_tuple(this->children[i]->getvisits(), this->children[i]->getmove(), this->children[i]));
	}
	return out;
}
void treestate::mark_keep(){
	this->keep = true;
	this->parent = NULL;
}
void treestate::del_inf_mut(){
	delete this->infer_mutex;
}
treestate::~treestate(){
	delete this->choose_mutex;
	delete this->finish_mutex;
	delete this->back_mutex;
	for(int i = 0;i<this->children.size();++i){
		if(!this->children[i]->keep){
			delete this->children[i];
		}
	}
}
