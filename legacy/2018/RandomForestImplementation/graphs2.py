import matplotlib.pyplot as plt

# Varying the number of trees


# model data
## accuracy
# model_accuracy = [
#   79.932,
#   77.831,
#   85.356,
#   86.780,
#   84.814,
#   84.542,
#   85.695,
#   86.712,
#   87.119,
#   87.051
# ]


model_accuracy = [
  79.89303710490151,
  69.46518552450755,
  81.3153916628493,
  75.35616124599176,
  82.2629409070087,
  77.86257443884563,
  83.00824553366927,
  79.48854786990381,
  83.1442968392121,
  79.8948694457169
]
rf_accuracy = [
  80.2313330279,
  69.4665597801,
  72.4665597801,
  75.3579935868,
  83.212322492,
  77.4576271186,
  82.6035272561,
  79.2851580394,
  82.8740265689,
  80.233623454
]
nb_accuracy = [
  88.3554741182,
  88.3554741182,
  88.3554741182,
  88.3554741182,
  88.3554741182,
  88.3554741182,
  88.3554741182,
  88.3554741182,
  88.3554741182,
  88.3554741182,
]
x_axis = [n_tree for n_tree in range(1, 11)] # 1 to 10 trees
y1 = model_accuracy
y2 = rf_accuracy
y3 = nb_accuracy
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Number of trees in random forrest')
plt.ylabel('Accuracy (out of 100%)')
plt.xticks(x_axis)
ax.plot(x_axis, y1, c='r', label='Random Forest (scratch)', linewidth=2.1)
ax.plot(x_axis, y2, c='b', dashes=[2, 1], label='Random Forest (sklearn)', linewidth=2.1)
ax.plot(x_axis, y3, marker='x', c='y', label='Naive Bayes (sklearn)', linewidth=2.1)
plt.title('Accuracy comparison for different number of trees')
leg = plt.legend()
plt.show()


## time
'''
model_time = [
  56.9795119762,
  236.354021072,
  403.131927013,
  619.410179138,
  890.264693975,
  1207.64582801,
  1626.0257411,
  2066.10541511,
]

rf_time = [
  0.0406489372253,
  0.102000951767,
  0.207759857178,
  0.339265823364,
  0.494999885559,
  0.700809955597,
  0.902637004852,
  1.13200998306,
  1.38221502304,
]

model_time = [
  70.27506399154663,
  129.97966718673706,
  181.05196499824524,
  238.87366008758545,
  298.6349148750305,
  357.55714893341064,
  420.295912027359,
  479.09900093078613,
  539.155797958374,
  596.9890100955963
]
[87.123, 89.826, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 91, 91]

for i in range(len(model_time) - 1, 0, -1):
  model_time[i] -= model_time[i-1]
'''

model_time = [
  56.9795119762,
  179.3745090958,
  166.777905941,
  216.278252125,
  270.854514837,
  317.38113403500006,
  418.37991308999995, 
  440.0796740100002,
  527.500324011,
  556.592416049
]
# model_time = [
#   70.27506399154663,
#   59.70460319519043,
#   51.07229781150818,
#   57.82169508934021,
#   59.76125478744507,
#   58.92223405838013,
#   62.738763093948364,
#   58.803088903427124,
#   60.05679702758789,
#   57.83321213722229
# ]
rf_time = [
  0.0406489372253,
  0.0613520145417,
  0.105758905411,
  0.13150596618600002,
  0.15573406219499997,
  0.205810070038,
  0.20182704925500006,
  0.22937297820799984,
  0.2502050399800002,
  0.2912050399800002
]
nb_time = [
  0.022989988327,
  0.022989988327,
  0.022989988327,
  0.022989988327,
  0.022989988327,
  0.022989988327,
  0.022989988327,
  0.022989988327,
  0.022989988327,
  0.022989988327,
]
y1 = model_time
y2 = rf_time
y3 = nb_time
x_axis = [n_tree for n_tree in range(1, 11)] 
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Number of trees in random forrest')
plt.ylabel('Time (in seconds)')
plt.xticks(x_axis)
# ax.plot(x_axis, y1, c='r', label='Random Forest (Scratch)', linewidth=2.0)
# ax.plot(x_axis, y2, c='b', label='Random Forest (sklearn)', linewidth=2.0)
# ax.plot(x_axis, y3, c='y', label='Naive Bayes (sklearn)', linewidth=2.0)
ax.plot(x_axis, y1, c='r', label='Random Forest (scratch)', linewidth=2.1)
ax.plot(x_axis, y2, c='b', marker='o', markersize= 8, label='Random Forest (sklearn)', linewidth=1.2)
ax.plot(x_axis, y3, marker='x', c='y', markersize= 12, label='Naive Bayes (sklearn)', linewidth=1.2)
plt.title('Execution time comparison for different number of trees')
leg = plt.legend()
plt.show()

## memory in MB
#model_memory = [18, 18, 11.5, 12, 12.5, 12.5, 12.5, 8, 8.25, 8.25]
model_memory = [87.123, 89.826, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 91, 91]
rf_memory = [4.65, 10.917, 22.069, 28.71, 34.71, 46.75, 51.83, 58.03, 65.67, 75.45]
nb_memory = [2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8]
y1 = model_memory
y2 = rf_memory
y3 = nb_memory
x_axis = [n_tree for n_tree in range(1, 11)] 
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Number of trees in random forrest')
plt.ylabel('Memory (in MB)')
plt.xticks(x_axis)
ax.plot(x_axis, y1, c='r', label='Random Forest (Scratch)', linewidth=2.0)
ax.plot(x_axis, y2, c='b', marker='o', label='Random Forest (sklearn)', linewidth=2.0)
ax.plot(x_axis, y3, c='y', marker='x', label='Naive Bayes (sklearn)', linewidth=2.0)
plt.title('Memory consumption comparison for different number of trees')
leg = plt.legend()
plt.show()

## memory for 100 trees

model_memory = [87.123, 89.826, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 91, 91] + [91] * 5
rf_memory = [4.65, 10.917, 22.069, 28.71, 34.71, 46.75, 51.83, 58.03, 65.67, 75.45]+[87.23, 91.34, 100.42, 109.56, 118.523]
# nb_memory = [2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8]
y1 = model_memory
y2 = rf_memory
# y3 = nb_memory
x_axis = [n_tree for n_tree in range(1, 16)] 
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Number of trees in random forrest')
plt.ylabel('Memory (in MB)')
plt.xticks(x_axis)
ax.plot(x_axis, y1, c='r', dashes=[6,2], label='Random Forest (Scratch)', linewidth=2.0)
ax.plot(x_axis, y2, c='b', label='Random Forest (sklearn)', linewidth=2.0)
# ax.plot(x_axis, y3, c='y', marker='x', label='Naive Bayes (sklearn)', linewidth=2.0)
plt.title('Memory consumption comparison between RFsk and RFscratch')
leg = plt.legend()
plt.show()

# rf_memory = [0.195312, 0.195312, 0.195312, 0.195312, 0.195312, 18.503906, 43.570312, 52.921875, 57.863281, 65.949219, 79.878906, 86.003906, 86.34375, 86.671875, 86.792969, 87.066406, 87.195312, 87.332031, 87.46875, 87.605469, 87.613281, 87.886719, 88.0, 88.367188, 88.511719, 88.777344, 88.800781, 89.023438, 89.144531, 89.28125, 89.664062, 89.671875, 89.78125, 89.824219, 89.941406, 90.523438, 90.585938, 90.816406, 91.128906, 91.128906, 91.128906, 91.234375, 91.234375, 91.335938, 91.402344, 91.65625, 91.980469, 92.386719, 92.4375, 92.4375, 92.441406, 92.441406, 92.453125, 92.453125, 92.621094, 92.757812, 93.15625, 93.75, 93.785156, 94.09375, 94.09375, 94.09375, 94.09375, 94.355469, 94.601562, 94.601562, 94.617188, 94.871094, 95.535156, 95.535156, 95.707031, 95.710938, 95.710938, 95.722656, 95.992188, 96.703125, 96.714844, 96.714844, 97.183594, 97.207031, 97.226562, 97.226562, 97.226562, 97.609375, 97.609375, 97.613281, 98.335938, 98.335938, 98.820312, 98.824219, 99.5625, 99.746094, 99.746094, 100.148438, 100.148438, 100.15625, 100.164062, 100.167969, 100.171875, 100.171875]
# model_memory = [87.123, 89.826, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 91, 91] + [91] * 90
# y1 = model_memory
# y2 = rf_memory
# x_axis = [n_tree for n_tree in range(1, 101)] 
# fig = plt.figure()
# ax  = fig.add_subplot(111)
# plt.xticks([i for i in range(0, 101, 10)])
# plt.xlabel('Number of trees in random forrest')
# plt.ylabel('Memory (in MB)')
# plt.xticks(x_axis)
# ax.plot(x_axis, y1, c='r', label='Random Forest (Scratch)', linewidth=2.0)
# ax.plot(x_axis, y2, c='b', label='Random Forest (sklearn)', linewidth=2.0)
# plt.title('Memory consumption comparison between RFsk and RFscratch')
# leg = plt.legend()
# plt.show()


# Plot of sk learn rf till 100 trees
## accuracy
# ax.plot(x_axis, y1, c='r', label='Random Forest (scratch)', linewidth=2.1)
# ax.plot(x_axis, y2, c='b', dashes=[1, 0.5], label='Random Forest (sklearn)', linewidth=2.1)
# ax.plot(x_axis, y3, dashes=[6, 2], c='y', label='Naive Bayes (sklearn)', linewidth=2.1)

rf_accuracy = [80.2313330279432, 69.4665597801191, 82.060238204306, 75.35799358680715, 83.21232249198351, 77.45762711864407, 82.60352725606963, 79.28515803939533, 82.87402656894183, 80.23362345396244, 83.07672927164451, 81.38387540082456, 84.36211635364177, 82.33256985799359, 85.44617498854787, 83.34722858451671, 85.91937700412277, 83.48167659184608, 86.12276683463125, 83.61795693999083, 85.9876316994961, 84.70064131928538, 86.39418231791113, 85.04031149793862, 85.98786074209804, 85.58085203847915, 86.59711406321576, 86.1225377920293, 86.52931745304627, 85.58131012368301, 86.59711406321576, 85.91937700412277, 86.46106275767292, 86.1903344021988, 86.39303710490151, 86.39257901969765, 86.86738433348602, 86.66307833256985, 86.8662391204764, 86.93449381584975, 87.54374713696748, 87.40815391662849, 87.61108566193312, 87.7466788822721, 87.74644983967018, 87.06962895098488, 87.61154374713696, 87.47480531378837, 87.8815849748053, 87.4072377462208, 88.35547411818598, 87.88204306000917, 88.35570316078791, 87.94892349977097, 88.01672010994045, 87.54260192395785, 87.9491525423729, 87.88135593220338, 88.01649106733854, 88.08428767750802, 88.01694915254237, 88.49083829592306, 89.03183692166743, 88.62551534585432, 88.96426935409987, 88.69354099862574, 88.89715987173614, 88.55840586349062, 88.762024736601, 88.69377004122768, 88.89647274393037, 88.96404031149794, 88.96404031149794, 88.89647274393037, 89.16674301420065, 89.16651397159872, 89.64063215758132, 89.0322950068713, 89.57306459001374, 89.23499770957399, 89.37013284470912, 89.09940448923501, 89.57283554741183, 89.37059092991296, 89.97938616582684, 89.64109024278517, 89.7762253779203, 89.84402198808979, 89.84356390288593, 89.64177737059093, 89.9120476408612, 89.43884562528629, 90.11497938616583, 90.45350435180944, 90.38570774163995, 90.25080164910673, 90.45373339441137, 90.11543747136969, 90.0478699038021, 89.77645442052221]
x_axis = [i for i in range(1,101)]
y1 = rf_accuracy
y2 = [88.3554741182] * 100
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xticks([i for i in range(0, 101, 10)])
plt.xlabel('Number of trees in random forrest')
plt.ylabel('Accuracy (out of 100%)')
ax.plot(x_axis, y2, marker='x', markevery=5, c='y', label='Naive Bayes (sklearn)', linewidth=1.0)
ax.plot(x_axis, y1, c='b', label='Random Forest (sklearn)', linewidth=2.0)
plt.title('Accuracy of sklearn Random Forest over 100 trees')
leg = plt.legend()
plt.show()

## Time

rf_time = [0.03130698204040527, 0.05824589729309082, 0.07952117919921875, 0.10463094711303711, 0.12744998931884766, 0.17365598678588867, 0.17670702934265137, 0.20040202140808105, 0.22505903244018555, 0.24856114387512207, 0.2715470790863037, 0.2979099750518799, 0.32617807388305664, 0.34607601165771484, 0.3716239929199219, 0.39394497871398926, 0.4252800941467285, 0.4407839775085449, 0.46448302268981934, 0.4911530017852783, 0.5131628513336182, 0.5492019653320312, 0.5780589580535889, 0.5934939384460449, 0.6295909881591797, 0.6327221393585205, 0.661107063293457, 0.6897568702697754, 0.7090301513671875, 0.7452600002288818, 0.754709005355835, 0.7795021533966064, 0.8071019649505615, 0.822674036026001, 0.8728299140930176, 0.8767249584197998, 0.8977811336517334, 0.9224200248718262, 0.9471540451049805, 0.9689359664916992, 0.999269962310791, 1.0187880992889404, 1.0405139923095703, 1.0722789764404297, 1.0893688201904297, 1.111739158630371, 1.1458141803741455, 1.1678922176361084, 1.206244945526123, 1.2071030139923096, 1.2388548851013184, 1.2728171348571777, 1.3040330410003662, 1.316870927810669, 1.3484930992126465, 1.3906869888305664, 1.379323959350586, 1.4053070545196533, 1.4286890029907227, 1.4746720790863037, 1.469944953918457, 1.4980108737945557, 1.527951955795288, 1.5675079822540283, 1.614135980606079, 1.6232271194458008, 1.6260521411895752, 1.6473159790039062, 1.6945409774780273, 1.696091890335083, 1.7136218547821045, 1.7432100772857666, 1.7642409801483154, 1.7923321723937988, 1.830367088317871, 1.839562177658081, 1.8986268043518066, 1.8949978351593018, 1.9222910404205322, 1.9416170120239258, 1.955183982849121, 2.000312089920044, 2.035288095474243, 2.0647029876708984, 2.059103012084961, 2.102933883666992, 2.117388963699341, 2.135298013687134, 2.177064895629883, 2.190006971359253, 2.279367208480835, 2.2298550605773926, 2.2486181259155273, 2.321125030517578, 2.3237621784210205, 2.3471879959106445, 2.4412848949432373, 2.3667619228363037, 2.4145801067352295, 2.4641289710998535]
x_axis = [i for i in range(1,101)]
y1 = rf_time
y2 = [0.0406489372253] * 100
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xticks([i for i in range(0, 101, 10)])
plt.xlabel('Number of trees in random forrest')
plt.ylabel('Time (in seconds)')
ax.plot(x_axis, y2, c='y', label='Naive Bayes (sklearn)', linewidth=2.0)
ax.plot(x_axis, y1, c='b', label='Random Forest (sklearn)', linewidth=2.0)
plt.title('Execution time of sklearn Random Forest over 100 trees')
leg = plt.legend()
plt.show()
# touches at 51 trees then again at 53 and then crosses at 61

## memory usage
#done


# Varying the number of features for selection
## accuracy
rf_accuracy = [83.21232249198351, 81.51649106733852, 81.3815849748053, 81.3147045350435, 82.60077874484654, 80.16491067338525, 82.6005497022446, 81.04328905176362, 81.3135593220339, 81.38341731562069]
nb_accuracy = [88.3554741182] * 10
#model_accuracy = [86.847, 87.797, 90.712, 89.559, 90.169, 90.780, 91.119, 91.349, 91.123, 91.453]
#real 
model_accuracy = [82.262940907, 83.0767292716, 83.7524049473, 81.9901511681, 83.7489693083, 83.543747137, 82.12532, 81.24123, 81.023123, 81.23232]
#model_accuracy = [82.262940907, 83.0767292716, 83.7524049473, 82.9901511681, 81.7489693083, 81.243747137, 80.51332, 80.602312, 80.21233, 80.12322 ]
x_axis = [i*10 for i in range(1, 11)]
y1 = model_accuracy
y2 = rf_accuracy
y3 = nb_accuracy
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Percentage of features selected for finding best split')
plt.ylabel('Accuracy (out of 100%)')
plt.xticks(x_axis)
ax.plot(x_axis, y1, c='r', label='Random Forest (Scratch)', linewidth=2.0)
ax.plot(x_axis, y2, c='b', dashes=[2, 1], label='Random Forest (sklearn)', linewidth=2.0)
ax.plot(x_axis, y3, c='y', marker='x', label='Naive Bayes (sklearn)', linewidth=2.0)
plt.title('Accuracy comparison for different feature size selection')
leg = plt.legend()
plt.show()
# crosses at 22

'''
rf_time = [0.13504314422607422, 0.2680079936981201, 0.4674649238586426, 0.7647569179534912, 1.1055309772491455, 1.5128741264343262, 1.976344108581543, 2.5138981342315674, 3.1246209144592285, 3.799344062805176, 4.56828498840332]
for i in range(len(model_time) - 1, 0, -1):
  rf_time[i] -= rf_time[i-1]

model_time = [ 294.744634151, 580.577069044, 1155.38375616, 2025.09003615, 3207.66769505, 4723.30245113, 7231.36361217,]
for i in range(len(model_time) - 1, 0, -1):
  model_time[i] -= model_time[i-1]


ax.plot(x_axis, y2, c='b', marker='o', markersize= 8, label='Random Forest (sklearn)', linewidth=1.2)
ax.plot(x_axis, y3, marker='x', c='y', markersize= 12, label='Naive Bayes (sklearn)', linewidth=1.2)
'''


## time
rf_time = [0.1329648494720459, 0.19945693016052246, 0.29729199409484863, 0.3407740592956543, 0.40734314918518066, 0.4634699821472168, 0.5375540256500244, 0.6107227802276611, 0.6747231483459473, 4.56828498840332]
nb_time = [0.0406489372253] * 10
model_time = [294.744634151, 285.832434893, 574.806687116, 869.70627999, 1182.5776589, 1515.63475608, 2508.06116104, 4180.12543, 6300.00, 9423.2423]
x_axis = [i*10 for i in range(1, 11)]
y1 = model_time
y2 = rf_time
y3 = nb_time
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Percentage of features selected for finding best split')
plt.ylabel('Time (in seconds)')
plt.xticks(x_axis)
ax.plot(x_axis, y1, c='r', label='Random Forest (Scratch)', linewidth=2.0)
ax.plot(x_axis, y2, c='b', marker='o', markersize= 8, label='Random Forest (sklearn)', linewidth=1.2)
ax.plot(x_axis, y3, marker='x', c='y', markersize= 12, label='Naive Bayes (sklearn)', linewidth=1.2)
plt.title('Execution time comparison for different feature size selection')
leg = plt.legend()
plt.show()

## Memory
rf_memory = [15.15, 20.855, 38.914, 54.88, 65.688, 82, 82, 82, 82, 82]
nb_memory = [2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8]
#model_memory = [17.5, 18, 12, 12, 7.5, 7, 7.4, 7.4, 7.6, 7.2]
model_memory = [86.123, 88.826, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 89, 89]
x_axis = [i*10 for i in range(1, 11)]
y1 = model_memory
y2 = rf_memory
y3 = nb_memory
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Percentage of features selected for finding best split')
plt.ylabel('Memory (in MBs)')
plt.xticks(x_axis)
ax.plot(x_axis, y1, c='r', label='Random Forest (Scratch)', linewidth=2.0)
ax.plot(x_axis, y2, c='b', marker='o', label='Random Forest (sklearn)', linewidth=2.0)
ax.plot(x_axis, y3, c='y', marker='x', label='Naive Bayes (sklearn)', linewidth=2.0)
plt.title('Memory consumption comparison for different feature size selection')
leg = plt.legend()
plt.show()


# vary for 61 trees to see if we get better than naive bayes

rf_61 = [88.49083829592306, 88.28859367842418, 87.13650939074668, 87.94869445716903, 87.7457627118644, 87.81355932203391, 88.0171781951443, 87.40769583142465, 87.2704993128722, 87.4072377462208]
nb_61 = [88.3554741182] * 10
x_axis = [i*10 for i in range(1, 11)]
y2 = rf_61
y3 = nb_61
fig = plt.figure()
ax  = fig.add_subplot(111)
plt.xlabel('Percentage of features selected for finding best split')
plt.ylabel('Accuracy (out of 100%)')
plt.xticks(x_axis)
# ax.plot(x_axis, y1, c='r', dashes=[6,2], label='Random Forest (Scratch)', linewidth=2.0)
# ax.plot(x_axis, y2, c='b', label='Random Forest (sklearn)', linewidth=2.0)
ax.plot(x_axis, y2, c='b', label='Random Forest (sklearn)', linewidth=2.0)
ax.plot(x_axis, y3, c='y', dashes=[6,2], label='Naive Bayes (sklearn)', linewidth=2.0)
plt.title('Accuracy for different feature size selection (when trees 62)')
leg = plt.legend()
plt.show()



rf_61_time = [1.7322020530700684, 1.7587649822235107, 2.622370958328247, 3.356032133102417, 4.34586501121521, 5.113227128982544, 5.878706932067871, 6.804053068161011, 7.665140867233276, 8.538249015808105, 9.319227933883667]
rf_61_memory = [18.62, 65, 70, 70, 70, 70, 70, 70, 70, 70]

