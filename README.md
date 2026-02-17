Video2Data
==========


Ellipse2Data
-------------

- [ellipse2data](ellipse2data.py)

Example usage for filtered data:
`python ellipse2data.py /path/to/folder/ --cutoff 1.5 --fs 30`

Example for raw data (and signed differences):
`python ellipse2data.py /path/to/folder/ --raw --signed`

Example for filtered data with csv output:
`python ellipse2data.py /path/to/folder --output filename.csv --cutoff 1.5 --fs 30`


To-Do
-------

- [x] Plot x distance
- [x] Plot y distance
- [x] Plot absolute distance
- [x] Find ellipse extremes ([xd, yd], [xu, yu], [xl, yl], [xr, yr])
- [ ] Test reconstruction accuracy
- [x] More points == more accuracy?
- [x] Check dependencies
- [ ] Argparse


Directory structure
-------------------

```
Video2Dynamics/
|   README.MD
|   ellipse2data3.py
|   LICENSE
|   AI_disclaimer.md
|
|___output/
|   |   data4model.json
|   |___animation/
|   |___plot/

```


Dataset - camera motion
-----------------------

<details> <summary> Camera motion </summary>
    
1. Moving
2. Static
3. Static
4. Moving
5. Static
6. Static
7. Moving
8. Static
9. Static
10. Static
11. Static
12. Moving
13. Moving
14. Moving
15. Moving
16. Static
17. Moving
18. Moving
19. Static
20. Moving
21. Moving
22. Moving
23. Moving
24. Moving
25. Moving
26. Static
27. Moving
28. Moving
29. Moving
30. Static
31. Moving
32. Static
33. Static
34. Static
35. Static
36. Static
37. Static
38. Static
39. Static
40. Static
41. Static
42. Static
43. Static
44. Moving
45. Moving
46. Moving
47. Moving
48. Moving
49. Moving
50. Static
51. Static
52. Static
53. Moving
54. Moving
55. Moving
56. Moving
57. Static
58. Moving
59. Moving
60. Moving
61. Static
62. Moving
63. Moving
64. Static
65. Moving
66. Moving
67. Moving
68. Moving
69. Moving
70. Static
71. Moving
72. Moving
73. Moving
74. Moving
75. Moving
76. Static
77. Moving
78. Moving
79. Moving
80. Static
81. Static
82. Moving
83. Moving
84. Moving
85. Moving
86. Moving
87. Moving
88. Static
89. Moving
90. Static
91. Moving
92. Static
93. Static
94. Moving
95. Moving
96. Moving
97. Static
98. Static
99. Moving
100. Moving

</details>

Notes
-----

-  argparse options: single frame, all frames, plot, animate.
    - Default: all frames, no animation, no plot, print results.

- how to test reconstruction accuracy?
