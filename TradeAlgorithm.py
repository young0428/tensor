def calc_score(data_dic,cur_index,action,possess,BATCH_SIZE):
	reward = []
	GAMMA = 0.01
	commision = 0.0015

	for i in range(BATCH_SIZE):
		score = 0
		if (possess[i] > 0) and (action[i] == 1):
			gap = data_dic[cur_index[i]][1]*(1-commision) - data_dic[possess[i]][1]*(1+commision)
			if gap < 0:
				score =  GAMMA*((1 + (cur_index[i] - possess[i])/30))*gap
			else:
				score = GAMMA*gap*( 1 - ((cur_index[i] - possess[i])/30))
			possess[i] = 0
		elif (possess[i] == 0) and (action[i] == 0) :
			possess[i] = cur_index[i]

		reward.append(score)

	return reward,possess

