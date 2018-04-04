class Evaluator:
    def Difference(self, actual, estimate):  #returns both the list of differences for all values, and a list of length 7 of the average difference for each 7 attributes
        #print("\nLENGTHS\n")
        #print(len(actual))
        #print(len(estimate))
        #print(len(actual[0]))
        diff = list()
        avg = [0,0,0,0,0,0,0]
        maxdiff = [0,0,0,0,0,0,0]
        for i in range(len(actual)):
            diff.append([])
            for j in range(len(actual[0])):
                diff[i].append(abs(actual[i][j] - estimate[i][j]))
                avg[j] = avg[j] + diff[i][j]
                if diff[i][j] > maxdiff[j]:
                    maxdiff[j] = diff[i][j]
        for i in range(7):
            avg[i] = avg[i]/len(actual)
        return diff, avg, maxdiff
