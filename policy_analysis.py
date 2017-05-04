def basic_output(values, policy):
    for state in values.keys():
        if state[0] <= 21 and not state[3] and policy[state] != -1:
            print("State:%s Value:%.4f PolicyAction:%d" % (state, values[state], policy[state]))


def policy_visualization(values, policy):
    for usable_ace in [True, False]:
        if usable_ace:
            player_range = range(21,11,-1)
        else:
            player_range = range(21,10,-1)
        for player_value in player_range:
            print("%d: " % player_value,  end="")
            for dealer_value in range(2,12):
                to_print = "%s" % (policy[(player_value, usable_ace, dealer_value, False)])
                if to_print == "-1":
                    to_print = "-"
                print(to_print, end="")
            print()
        print("    234567890A")
        print()
