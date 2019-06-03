
#------------------------------------------
#init hidden markov state
#------------------------------------------
model = HiddenMarkovModel()



#assign probability for the observations
words = read_observations("words.txt")
emission_prob = DiscreteDistribution(  assign_default_prop(words) )


#---------------------------------------------------
#creating states
#---------------------------------------------------
print("creating states...",end="")
states = create_default_state( emission_prob ,hidden_states)
print(states)
print("ok")

#------------------------------------------
#add hidden states
#------------------------------------------
print("add states...",end="")
model.add_states(states)
print("ok")

#-------------------------------------
#add transition matrix
#-------------------------------------

set_default_transition(model,states)

#-----------------------------------------------
#bake the model
#-----------------------------------------------
print("baking model...", end="")
model.bake()
print("ok")
#print("-----------------\n\n\n-----------------\n"+ model.to_json())



text = "io mangio tante mele"

seq = numpy.array( tokenize(text) )
hmm_predictions = model.predict(seq)
res = list()
for x in hmm_predictions:
    res.append(hidden_states[x] )

print ("sequence: {}".format(' '.join(seq)))
print ("hmm pred: {}".format(' '.join(res)))