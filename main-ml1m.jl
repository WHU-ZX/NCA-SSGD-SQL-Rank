# version 12.0:
# 1. Rewrite gradient calculation to reduce time complexity to O(d2_bar)
# 2. Remove line search and calculate average of gradients


# main("ml1m_oc_50_train_ratings.csv", "ml1m_oc_50_test_ratings.csv", 0.2, 0.9, 1, 4, 100, 3)


# train 		: 训练集数据文件名
# test  		: 测试集数据文件名
# learning_rate : 学习率
# decay_rate 	: 学习率减少率
# T				: 每过T个iteration就将learning_rate乘以decay_rate
# lambda		: 正则化参数，在论文中的损失函数中出现
# r				: U、V的维度，即假设的X的秩的最大值
# ratio			: (0:1)的比值
function main(train, test, learning_rate, decay_rate, T, lambda, r, ratio, io)
	#train = "ml1m_oc_50_train_ratings.csv"
	#test = "ml1m_oc_50_test_ratings.csv"
	# requires ratio to be integer, usually 3 works best
	X = readdlm(train, ',' , Int64); # 读训练集
	x = vec(X[:,1]);	# 表格数据第一列	表示user
	y = vec(X[:,2]);	# 表格数据第二列	表示item
	v = vec(X[:,3]);	# 表格数据第三列	表示rating
	Y = readdlm(test, ',' , Int64);	# 读测试集
	xx = vec(Y[:,1]);
	yy = vec(Y[:,2]);
	vv = vec(Y[:,3]);
	n = max(maximum(x), maximum(xx)); 		# 用户个数n
	msize = max(maximum(y), maximum(yy));	# item个数m

	# 创建稀疏矩阵
	X = sparse(x, y, v, n, msize); # userid by movieid		x，y表示矩阵下标,v表示矩阵的值
	Y = sparse(xx, yy, vv, n, msize);
	# julia column major 
	# now moveid by userid
	# 稀疏矩阵是按列下标递增排序的，列下标相同时，按行下标递增排列
	X = X'; 
	Y = Y'; 
	rows = rowvals(X);	# 返回行的索引
	vals = nonzeros(X);	# 如果c代表列索引，则X中，(rows(i),c(i)) = vals(i)  对任意i
	cols = zeros(Int, size(vals)[1]);	# 创建长度和vals一样的全为0的向量
	index = zeros(Int, n + 1);	# 创建长度为n+1的全0向量   标记原矩阵中每一列的开始位置

	d2, d1 = size(X);	# d2是原X的列数，d1是原X的行数
	cc = 0;
	# need to record new_index based on original index	-- 需要根据原始索引记录新的索引
	# such that later, no need to shift each iteration, only need to swap the zero part	-- 这样以后不需要移动每个迭代，只需要交换零的部分
	new_len = 0; # 新的各个列的长度之和
	new_index = zeros(Int, d1 + 1); # 标记新矩阵中每一列的开始位置的
	new_index[1] = 1;
	for i = 1:d1	# 从现X的1到d1列
		index[i] = cc + 1;	# index[i]表示原本第i列的第一个元素的行标在rows中的index
		tmp = nzrange(X, i);	# 返回X在rows = rowvals(X)的rows中的索引tmp，X(rows(tmp(k)),i)有元素  对任意k在tmp中均成立
		nowlen = size(tmp)[1];	# 第i列的非0元素个数，即打分不为0的个数
		newlen = nowlen * (1 + ratio);	# 扩展后的第i列的长度
		new_len += newlen;	# 增加总长度（将第i列的长度加入总长度中)
		new_index[i + 1] = new_index[i] + newlen;	# new_index[i]为原本第i列的第一个值在总向量（将矩阵每列长度修改后，按列stack成一个很长的向量）中的index
		for j = 1:nowlen	# 使得现矩阵X的第rows[k]行，第i列（cols[k]列）的值是vals[k]
			cc += 1;
			cols[cc] = i;
		end
	end
	index[d1 + 1] = cc + 1;
	# no need to sort for 0/1 data, so we don't need sort_input function in sqlrank10.jl 
	# ASSUMPTION: new_vals containing all 0's and 1's
	# we also need a function to shuffle all 1's and swapping 0's 
	new_rows = zeros(Int, new_len);	# 新的rows
	new_cols = zeros(Int, new_len);	# 新的cols
	new_vals = zeros(Int, new_len);	# 新的vals

	# 稀疏矩阵中没有的元素不一定是0，而此处代码将原本没有数据的地方随机地取一些置为0，使得新的X'的每一列的0：1 = ratio
	# 从下一段代码可以看出，新矩阵的行数等于原矩阵X'的行数
	for i = 1:d1
		rows_set = Set{Int}();
		for j = index[i]:(index[i + 1] - 1) # j从第i列的第一个元素在rows中的下标到第i列最后一个元素在rows中的下标
			push!(rows_set, rows[j]);
		end
		nowlen = new_index[i + 1] - new_index[i];
		nowOnes = div(nowlen, 1 + ratio);	# 目前1的个数
		# 这一段其实是使得原矩阵的第i行第j列与现矩阵的第i行第j列相同（非0元），只是希望以new_rows,new_cols,new_vals的形式暂时存储
		for j = 1:nowOnes	# 将原本第i列的所有非0值拷贝到新的第i列前面
			new_rows[new_index[i] + j - 1] = rows[index[i] + j - 1];
			new_cols[new_index[i] + j - 1] = i;
			new_vals[new_index[i] + j - 1] = vals[index[i] + j - 1];
		end
		nowStart = new_index[i] + nowOnes;	# 从上一段填充过的后面一个开始
		nowEnd = new_index[i + 1] - 1;	# 到第i列的最后一个元素在rouw中的下标结束
		# 此段要求原矩阵X'中每一列的 （无数据数 :1数） >= ratio，否则会陷入死循环
		for j = nowStart:nowEnd	# 此段将新矩阵的第i列原矩阵中为0的对应元素全置为0
			while true
				row_idx = rand(1:d2);	# 随机生成1到d2间的整数（都可以取到），d2表示X'的行数，即论文中的m
				if !(row_idx in rows_set)	# X'在第row_idx行第i列原元素为0则接着执行
					new_rows[j] = row_idx;
					new_cols[j] = i;
					new_vals[j] = 0.0;
					push!(rows_set, row_idx);
					break;
				end
			end  
		end
	end

	# 此段对测试集Y进行类似的操作，找出Y'的rows,cols,vals和index
	rows_t = rowvals(Y);
	vals_t = nonzeros(Y);
	cols_t = zeros(Int, size(vals_t)[1]);
	index_t =  zeros(Int, n + 1)
	cc = 0;
	for i = 1:d1
		index_t[i] = cc + 1;
		tmp = nzrange(Y, i);
		nowlen = size(tmp)[1];
		for j = 1:nowlen
			cc += 1
			cols_t[cc] = i
		end
	end
	index_t[d1 + 1] = cc + 1;

	# again, no need to sort rows_t, vals_t, cols_t under the ASSUMPTION
	# we also don't need levels
	
	srand(123456789);
	U = 0.1*randn(r, d1); 	# r = 100,是X的秩   生成元素满足标准正态分布的r*d1即r*n的矩阵
	V = 0.1*randn(r, d2);	# 生成元素满足标准正态分布的r*d2即r*m的矩阵
    #U = rand(r,d1)*(0.1 - (-0.1)) + (-0.1)
    #V = rand(r,d2)*(0.1 - (-0.1)) + (-0.1)
	m = comp_m(new_rows, new_cols, U, V);
	# no need to calculate max_d_bar, since we are using all 1's and 0's appended of ratio 1:ratio 
	println("rank: ", r, ", ratio of 0 vs 1: ", ratio, ", lambda:", lambda, ", learning_rate: ", learning_rate);
	write(io,"rank: ", string(r), ", ratio of 0 vs 1: ", string(ratio), ", lambda:", string(lambda), ", learning_rate: ", string(learning_rate),"\n");
	# no need for obtain_R method, but we do need a shuffling method I call it stochasticQueuing 
	# (that is why I name the method as sqlrank: stochastic queuing listwise ranking algorithm)
	
	println("iter time objective_function precision@K = 1, 5, 10");
	write(io,"iter time objective_function precision@K = 1, 5, 10\n");
	obj = objective(new_index, m, new_rows, d1, lambda, U, V);
	p1,p2,p3=compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t);
	println("[", 0, ",", obj, ", ", p1," ",p2," ",p3, "],");
	write(io,"[", string(0), ",", string(obj), ", ", string(p1)," ",string(p2)," ",string(p3), "],\n");
    #println("[", 0, ",", obj, "],");

	totaltime = 0.00000;
	num_epoch = 121;
    num_iterations_per_epoch = 1;
	nowobj = obj;
	m1 = 0;
	m2 = 0;
	m3 = 0;
	for epoch = 1:num_epoch
		tic();
		for iter = 1:num_iterations_per_epoch
			U, m = obtain_U(new_rows, new_cols, new_index, U, V, learning_rate, d1, r, lambda);
			V = obtain_V(new_rows, new_cols, new_index, m, U, V, learning_rate, d1, r, lambda);
		end
        
        new_rows = stochasticQueuing(new_rows, new_index, d1, d2, ratio);
        
        totaltime += toq();
	    #if (epoch - 1) % 3 == 0
        #    learning_rate = learning_rate * 0.3
        #end
        #learning_rate = learning_rate * 0.95
        if (epoch - 1) % T == 0
            learning_rate = learning_rate * decay_rate
			p1,p2,p3=compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t);
			if p1 > m1
				m1 = p1;
			end
			if p2 > m2
				m2 = p2;
			end
			if p3 > m3
				m3 = p3;
			end
		    m = comp_m(new_rows, new_cols, U, V);
		    nowobj = objective(new_index, m, new_rows, d1, lambda, U, V);
			println("[", epoch, ", ", totaltime, ", ", nowobj, ", ", p1,", ",p2,", ",p3, "],");
			write(io,"[", string(epoch), ", ", string(totaltime), ", ", string(nowobj), ", ", string(p1),", ",string(p2),", ",string(p3), "],\n");
	    else
            m = comp_m(new_rows, new_cols, U, V);
            nowobj = objective(new_index, m, new_rows, d1, lambda, U, V);
			println("[", epoch, ", ", totaltime, ", ", nowobj);
			write(io,"[", string(epoch), ", ", string(totaltime), ", ", string(nowobj),"\n")
        end
	end
	return m1, m2, m3
end

function stochasticQueuing(rows, index, d1, d2, ratio) # 重新执行SQ Procees得到new_rows
	new_rows = zeros(Int, size(rows)[1]);
	for i = 1:d1
		nowlen = index[i + 1] - index[i];
		nowOnes = div(nowlen, 1 + ratio);
		newOrder = shuffle(1:nowOnes);
		rows_set = Set{Int}();
		for j = 1:nowOnes
			oldIdx = index[i] + j - 1;
			row_j = rows[oldIdx];
			push!(rows_set, row_j);
			newIdx = index[i] + newOrder[j] - 1;
			new_rows[newIdx] = row_j;
		end
		nowStart = index[i] + nowOnes;
		nowEnd = index[i + 1] - 1;
		for j = nowStart:nowEnd
			while true
				row_idx = rand(1:d2);
				if !(row_idx in rows_set)
					new_rows[j] = row_idx;
					push!(rows_set, row_idx);
					break;
				end
			end  
		end
	end
	return new_rows	
end

function obtain_U(rows, cols, index, U, V, s, d1, r, lambda)
	m = comp_m(rows, cols, U, V);
	grad_U = comp_gradient_U(rows, cols, index, m, U, V, s, d1, r, lambda);
	U = U - s * grad_U;
	m = comp_m(rows, cols, U, V);
	return U, m
end

function comp_gradient_U(rows, cols, index, m, U, V, s, d1, r, lambda)
	grad_U = zeros(size(U));
	for i = 1:d1
		d_bar = index[i+1] - index[i];
		grad_U[:,i] = comp_gradient_ui(rows, cols, index, d_bar, m, i, V, r);
	end
	grad_U += 2*lambda/3 * U;
	return grad_U
end

function comp_gradient_ui(rows, cols, index, d_bar, m, i, V, r)
	cc = zeros(d_bar);
	tt = 0.0;
	total = 0.0;
	for t = d_bar:-1:1
		tmp = m[index[i] - 1 + t];
		total += exp(tmp);
		tt += 1 / total;
	end
	total = 0.0;
	for t = d_bar:-1:1
		ttt = m[index[i] - 1 + t];
		cc[t] -= ttt * (1 - ttt);
		cc[t] += exp(ttt) * ttt * (1 - ttt) * tt;
		total += exp(ttt);
		tt -= 1 / total;
	end

	res = zeros(r);
	for t = 1:d_bar
		res += cc[t] * V[:,rows[index[i] - 1 + t]];
	end
	return res
end

function obtain_V(rows, cols, index, m, U, V, s, d1, r, lambda)
	grad_V = comp_gradient_V(rows, cols, index, m, U, V, s, d1, r, lambda);
	V = V - s * grad_V;
	return V
end

function comp_gradient_V(rows, cols, index, m, U, V, s, d1, r, lambda) 
	grad_V = zeros(size(V));
	for i = 1:d1
		d_bar = index[i+1] - index[i];
		cc = zeros(d_bar);
		tt = 0.0;
		total = 0.0;
		for t = d_bar:-1:1
			tmp = m[index[i] - 1 + t];
			total += exp(tmp);
			tt += 1 / total;
		end
		total = 0.0;
		for t = d_bar:-1:1
			ttt = m[index[i] - 1 + t];
			cc[t] -= ttt * (1 - ttt);
			cc[t] += exp(ttt) * ttt * (1 - ttt) * tt;
			total += exp(ttt);
			tt -= 1 / total;
		end

		for t = 1:d_bar
			j = rows[index[i] - 1 + t]
			grad_V[:,j] += cc[t] * U[:,i]
		end	
	end
	F = svd(V);
	grad_V += lambda * 2/3 * F[1] * F[3]';
	return grad_V
end

function logit(x)
	return 1.0/(1+exp(-x))
end

function comp_m(rows, cols, U, V)
	m = zeros(length(rows));
	for i = 1:length(rows)
		m[i] = logit(dot(U[:,cols[i]], V[:,rows[i]]));
	end
	return m
end

# 计算目标函数
# 原本的rows就是得分为1的在前面，为0的在后面，所以rows一定程度上就可以理解成Π
function objective(index, m, rows, d1, lambda, U, V) # log(fi(x)) = sigmoid(x)
	res = 0.0;
	# 下面代码计算损失函数中的f(U,V)
	for i = 1:d1 # 从1到d1列进行遍历（总共也就d1列）  ### 目标函数中的1到n求和
		tt = 0.0;
		d_bar = index[i+1] - index[i]; # d_bar表示第i列的长度（有元素的个数）
		# tt = X'的第i列的fi(X'[k,i])对k求和
		for t = d_bar:-1:1 # 从d_bar递减到1, rows[t]遍历了X'第i列的所有有元素的行下标
			# since we will shuffle new_rows, new_cols, and obtain new m
			# we don't need to shuffle again for m (ASSUMPTION: we only have 1's and 0's)
			tmp = m[index[i] - 1 + t]; # U、V确定的X'的第i列第rows[t]行被sigmoid函数作用的结果，即log(fi)
			tt += exp(m[index[i] - 1 + t]); # tt+=fi(X'[rows[t],i])
			res -= tmp;
			res += log(tt);
		end
	end
	res += lambda / 3 * (vecnorm(U) ^ 2 + 2 * unclear_norm(V));
	return res
end

function compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t)
	K = [1, 5, 10]; # K has to be increasing order
	precision = [0, 0, 0];	# 精度
	for i = shuffle(1:d1)[1:1000]	# shuffle函数将1到d1随机打乱
	#for i = 1:d1
		###############################这一段做的事###############################
		# 取X'[:,i]中的没有数据的值(行下标为j)用 (U'V)'[j,i] 替代，其他的用负无穷代替
		# 对于topk  (k = 1  5  10)，从c从1到k，找到X'[:,i]中第c大的值，记其下标为j
		# 若X'(替代后的)的第j行第i列是来自(U'V)'[j,i]，且Y'的第j行第i列有值，则topk的precision++
		# 将 topk的precision = precision/k/1000
		################################相当于#################################
		# 取X'[:,i]中的没有数据的值(行下标为j)用 (U'V)'[j,i] 替代，其他的用负无穷代替
		# 找到X'[:,i]前k大的值(不能是负无穷，否则跳过负无穷)，如果Y'矩阵中对应的元素存在，则topk的precision++
		# 将 topk的precision = precision/k/1000

		tmp = nzrange(Y, i); # 获得第测试集Y的第i列的行下标的集合 a:b
		test = Set{Int64}();
		for j in tmp
            push!(test, rows_t[j]);
		end
		#test = Set(rows_t[tmp])
		if isempty(test) # 如果测试集第i列没有值，则跳过，处理下一列
			continue
		end
		tmp = nzrange(X, i); # 获得第训练集X的第i列的行下标的集合 a:b
		vals_d2_bar = vals[tmp]; # 获得训练集第i列的值的集合 type:  [a,b,c,d]
		train = Set(rows[tmp]); # 将训练集第i列在rows中的下标存入集合train
		score = zeros(d2); # 创建长度为d2（训练集X'的行数）的零向量
		ui = U[:, i]; # U的第i列
		for j = 1:d2 # 如果训练集第j行有值，则将score[j]设为 很小的负数，否则设为 X'[j,i]（即X[i,j]）
			if j in train
				score[j] = -10e10;
				continue;
			end
			vj = V[:, j];
			score[j] = dot(ui,vj);
		end
		p = sortperm(score, rev = true); # p为使得score从大到小的index
		for c = 1: K[length(K)] # c从1到10   针对score中由X'[j,i]确定的值
			j = p[c]; # score中第c高的值的index
			if score[j] == -10e10 # 如果score中第c高的值是-10e10
				break;
			end
			if j in test # 如果测试集第i列第j行有值
				for k in length(K):-1:1 # k从3到1
					if c <= K[k]
						precision[k] += 1;
					else
						break;
					end
				end
			end
		end
	end
	#precision = precision./K/d1;
	precision = precision./K/1000;
	return precision[1], precision[2], precision[3]
end

function unclear_norm(M)
    F = svd(M);
    sigma = F[2];
    norm = sum(sigma);
end

# train, test, learning_rate, decay_rate, T, lambda, r, ratio
# 暂时的最好参数:lr=0.1,dr=0.9
# main("ml1m_oc_50_train_ratings.csv", "ml1m_oc_50_test_ratings.csv", 0.1, 0.9, 1, 4, 100, 3)

function testParameters()
	train = "ml1m_oc_50_train_ratings.csv";
	test = "ml1m_oc_50_test_ratings.csv";
	prefix = "res/ml1m/"
	learning_rate_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
	decay_rate_list = [0.75, 0.8, 0.85, 0.9, 0.95];
	lambdas = [3, 4, 5, 6, 7, 8, 9];
	T = 1;
	r = 100;
	ratio = 3;
	result_path = string(prefix,"result.txt");
	rio = open(result_path,"w");
	for lambda in lambdas
		for lr in learning_rate_list
			for dr in decay_rate_list
				println("-----------------------------------------------------------------------")
				file_name = string(prefix,string(lr),"_",string(dr),"_",string(T),"_",string(lambda),"_",string(r),"_",string(ratio),".txt")
				io = open(file_name,"w");
				m1, m2, m3 = main(train,test,lr,dr,T,lambda,r,ratio,io);
				write(rio,file_name, " : ",string(m1), " , ", string(m2),  " , ", string(m3), "\n");
			end
		end
	end
end

function findLambda()
	train = "ml1m_oc_50_train_ratings.csv";
	test = "ml1m_oc_50_test_ratings.csv";
	prefix = "res/test/"
	lambdas = [1,2,3,4,5,6,7,8,9,10,11,12];
	learning_rate = 0.1
	decay_rate = 0.9
	T = 1;
	r = 100;
	ratio = 3;
	result_path = string(prefix,"result.txt");
	rio = open(result_path,"w");
	
	for lambda in lambdas

		println("-----------------------------------------------------------------------")
		file_name = string(prefix,string(learning_rate),"_",string(decay_rate),"_",string(T),"_",string(lambda),"_",string(r),"_",string(ratio),".txt")
		io = open(file_name,"w");
		m1, m2, m3 = main(train,test,learning_rate,decay_rate,T,lambda,r,ratio,io);
		write(rio,file_name, " : ",string(m1), " , ", string(m2),  " , ", string(m3), "\n");
	end
end

testParameters();

# 得出lambda = 3-10的效果还可以
# findLambda();