def basic_output(values, policy):
    for state in values.keys():
        if state[0] <= 21 and not state[3] and policy[state] != -1:
            print("State:%s Value:%.4f PolicyAction:%d" % (state, values[state], policy[state]))
