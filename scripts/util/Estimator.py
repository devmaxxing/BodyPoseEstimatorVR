from util import transformations
class Estimator:
    def Estimate(self, data):  
        
        result = []
        for i in range(len(data)):
            result.append([])
            quaternion = (data[i][0], data[i][1], data[i][2], data[i][3])
            euler = transformations.euler_from_quaternion(quaternion)
            L = list(euler)
            L[0] = 0
            L[1] = 0
            newQuat = transformations.quaternion_from_euler(L[0], L[1], L[2])
            

            result[i].append(newQuat[0])
            result[i].append(newQuat[1])
            result[i].append(newQuat[2])
            result[i].append(newQuat[3])
            result[i].append(data[i][4])
            result[i].append((data[i][5])-5.2)
            result[i].append(data[i][6])
        return result
