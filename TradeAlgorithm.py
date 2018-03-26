def calc_score(data_dic,cur_index,action,possess,BATCH_SIZE,UPDATE_TERM):
	reward = []
	real_reward = []
	GAMMA = 0.00001
	commision = 0.00001
	change = []
	for i in range(BATCH_SIZE):
		score = 0
		real_reward.append(0)
		change = data_dic[cur_index[i]][1] - data_dic[cur_index[i]-UPDATE_TERM][1]
		if possess[i] > 0 :
			score += change*GAMMA
		if (possess[i] > 0) and (action[i] == 1):
			gap = data_dic[cur_index[i]][1]*(1-commision) - data_dic[possess[i]][1]*(1+commision)
			past_score = data_dic[cur_index[i]][1] - data_dic[possess[i]][1]
			real_reward[i] = gap
			if gap < 0:
				score += GAMMA*(gap - past_score)  #*(1+(cur_index[i] - possess[i])/60000)
			else:
				score += GAMMA*(gap - past_score)
			possess[i] = 0
		elif (possess[i] == 0) and (action[i] == 0) :
			possess[i] = cur_index[i]

		reward.append(score)

	return real_reward,reward,possess

