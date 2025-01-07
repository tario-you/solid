\ Model mean_variance_optimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2.281824691522146 x_0 ^2 - 0.0454161905235998 x_0 * x_1
   + 0.0743253788905428 x_0 * x_2 + 0.0293925581415543 x_0 * x_3
   + 2.663306738052048 x_1 ^2 + 0.0327022906843394 x_1 * x_2
   + 0.0558099430453188 x_1 * x_3 + 2.292197009104976 x_2 ^2
   + 0.2348506963671398 x_2 * x_3 + 2.584677209013344 x_3 ^2 ] / 2 
Subject To
 budget: x_0 + x_1 + x_2 + x_3 = 1
 target_return: 0.8472844066929068 x_0 + 3.524385027535183 x_1
   + 0.8440230494811587 x_2 + 1.261688082814821 x_3 >= 0.1
Bounds
End
