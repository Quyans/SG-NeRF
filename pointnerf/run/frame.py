poses = [

[[-0.98577744,-0.099129,-0.00645936,1.226757  ],
 [-0.27931246, 0.39184898,-0.87660706,  2.732021  ],
 [-0.01145575, -0.91423637, -0.4050196,   1.377719  ],
 [ 0,          0,          0,          1        ]],

[[-0.9760648,  -0.09825459,  0.13881285,  1.5208713 ],
 [-0.40493867,  0.38910225, -0.8274209,   2.8445752 ],
 [-0.07120963, -0.9156055,  -0.3957222,   1.3947518 ],
 [ 0.,          0.,          0.,          1.,        ]],

[[-0.9670942,  -0.08553391,  0.19758098,  1.7438469 ],
 [-0.44923216,  0.33406314, -0.8286094,   2.7600791 ],
 [-0.09217258, -0.9398431,  -0.32893693,  1.3298492 ],
 [ 0.,          0.,          0.,          1.,        ]],

[[-0.83897275, -0.12779735,  0.5112904,   1.6458061 ],
 [-0.7011031,   0.42898077, -0.56958866,  2.9750957 ],
 [-0.22741987, -0.89160067, -0.39157194,  1.4511441 ],
 [ 0.,          0.,          0.,          1.,        ]]

]

c2w=[]
time=[]


# c2w.append([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]])

# c2w.append([[1.,0,0,1.],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]])
for pose in poses:
    c2w.append(pose)

time.append(0.)
time.append(1.)