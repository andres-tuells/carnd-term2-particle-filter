/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

#define NUMBER_OF_PARTICLES 500 //number of particles to use
#define EPS 0.001


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).ç
	static default_random_engine gen;
	gen.seed(42);
	num_particles = NUMBER_OF_PARTICLES;
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	//resize particle and weight vectors to num_particles size
	particles.resize(num_particles);
	weights.resize(num_particles);

	double init_weight = 1.0/num_particles;//sum weights=1, uniform distribution

	for (unsigned int i = 0; i < num_particles; i++){
		particles[i].id = i;
		particles[i].x      = dist_x(gen);
		particles[i].y      = dist_y(gen);
		particles[i].theta  = dist_theta(gen);
		particles[i].weight = init_weight;
	}	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// computate constants
	const double vel_dt = velocity * delta_t;
	const double yaw_dt = yaw_rate * delta_t;
	const double vel_yaw = velocity/yaw_rate;

	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	for (unsigned int i = 0; i < num_particles; i++){
		if (fabs(yaw_rate) < EPS){ //no change of direction
			particles[i].x += vel_dt * cos(particles[i].theta);
			particles[i].y += vel_dt * sin(particles[i].theta);
		}else{
			const double theta_new = particles[i].theta + yaw_dt;
			particles[i].x += vel_yaw * (sin(theta_new) - sin(particles[i].theta));
			particles[i].y += vel_yaw * (-cos(theta_new) + cos(particles[i].theta));
			particles[i].theta = theta_new;
		}
		//Gaussian noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
	
		//current observation
		LandmarkObs o = observations[i];

		// set minimum distance to maximum possible
		double min_dist = numeric_limits<double>::max();

		//no id associated yet
		int id = -1;
	
		for (unsigned int j = 0; j < predicted.size(); j++) {
			//urrent prediction
			LandmarkObs p = predicted[j];
	  
			//distance between current and predicted landmarks
			double cur_dist = dist(o.x, o.y, p.x, p.y);

			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				id = p.id; //id of the nearest predicted landmark to observed landmark
			}
		}

		// set the observation's id to the nearest predicted landmark's id
		observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (unsigned int i = 0; i < num_particles; i++) {

		//particle coordinates
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		//landmark predictions
		vector<LandmarkObs> predictions;

		// for each map landmark...
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			//landmark coordinates and id
			float lm_x   = map_landmarks.landmark_list[j].x_f;
			float lm_y   = map_landmarks.landmark_list[j].y_f;
			int   lm_id  = map_landmarks.landmark_list[j].id_i;
	  
			//only add to predictions landmarks inside sensor range rectangle
			if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {
				predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
			}
		}

		//transform observations from car coordinates to map coordinates
		vector<LandmarkObs> map_observations;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
			double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
			int id = observations[j].id;
			map_observations.push_back(LandmarkObs{ id, t_x, t_y });
		}

		//associate data from predictions to mapped observations
		dataAssociation(predictions, map_observations);

		//restart weight
		particles[i].weight = 1.0;

		for (unsigned int j = 0; j < map_observations.size(); j++) {
	  
			//observation coordinates
			double o_x = map_observations[j].x;
			double o_y = map_observations[j].y;

			//prediction coordinates
			double p_x;
			double p_y;

			int associated_prediction_id = map_observations[j].id;

			//iterate predictions to find assiciated one
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == associated_prediction_id) {
					p_x = predictions[k].x;
					p_y = predictions[k].y;
					break;
				}
			}

		// calculate weight for this observation with multivariate Gaussian
		double s_x = std_landmark[0];
		double s_y = std_landmark[1];
		double o_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(p_x-o_x,2)/(2*pow(s_x, 2)) + (pow(p_y-o_y,2)/(2*pow(s_y, 2))) ) );

		particles[i].weight *= o_w;
	}
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
 	vector<Particle> new_particles;

  	//collect all particle weights
  	vector<double> weights;
  	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
  	}

    discrete_distribution<int> d(weights.begin(), weights.end());
  	for(unsigned i = 0; i < num_particles; i++){
    	auto index = d(gen);
    	new_particles.push_back(move(particles[index]));
  	}
  	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
