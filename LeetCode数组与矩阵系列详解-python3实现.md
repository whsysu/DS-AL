#### #数组首先想到：双指针+二分法+分治法+递归法===>所有的算法都是归纳总结

```python
#数组与矩阵

#283. 移动零,将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
#双指针交换法
def moveZeroes(self, nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    要求原地移动：
    数组首先想到：双指针
    思路：i指向第一个0元素，j指向i后面第一个非0元素
    """
    if not nums:
        return nums
    n = len(nums)
    i = j = 0
    while i < n:
        while i < n and nums[i] != 0:
            i += 1
        j = i + 1  # j在i后面
        while j < n and nums[j] == 0:
            j += 1
        if j < n:
            nums[i], nums[j] = nums[j], nums[i]
        else:
            return nums
    return nums
#双指针法移动法
def moveZeroes(self, nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    要求原地移动：
    数组首先想到：双指针
    思路：i指向第一个0元素，j指向i后面第一个非0元素
    """
    if not nums:
        return nums
    n = len(nums)
    index = 0
    for i in nums:
        if i != 0:
            nums[index] = i
            index += 1
    for i in range(index, n):
        nums[i] = 0
    return nums


#566. 重塑矩阵：将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。
def matrixReshape(self, nums, r: int, c: int):
    '''本质上：二维数组在内存中是连续的也就是说本质为一维数组'''
    row,col=len(nums),len(nums[0])
    if row*col!=r*c:
        return nums
    new=[[0 for i in range(c)] for j in range(r)]
    for i in range(r):
        for j in range(c):
            #i*c+j==r1*col+c1  =>求r1和c1
            new[i][j]=nums[(i*c+j)//col][(i*c+j)%col]
    return new


#485. 最大连续1的个数:给定一个二进制数组， 计算其中最大连续1的个数。
#双指针法
def findMaxConsecutiveOnes(self, nums) -> int:
    '''思路：数组首先想到双指针；
    最大连续串=》想到动态规划
    显然，本题使用双指针即可'''
    pre = last = 0
    res = 0
    n = len(nums)
    while pre < n:
        while pre < n and nums[pre] != 1:   #pre指向第一个1
            pre += 1
        last = pre
        while last < n and nums[last] == 1:  #last指向pre后面的第一个0
            last += 1
        res = max(res, last - pre)
        pre = last   #一定要更新pre
    return res
#这是什么法？？
def findMaxConsecutiveOnes(self, nums) -> int:
    res = cur = 0
    for i in nums:
        cur = 0 if i == 0 else cur + 1
        res = max(res, cur)
    return res

#240. 搜索二维矩阵 每行的元素从左到右升序排列；每列的元素从上到下升序排列。


#378. 有序矩阵中第K小的元素： n x n 矩阵，每行和每列按升序排序，找第 k 小的元素。
#请注意，它是排序后的第 k 小元素，而不是第 k 个不同的元素。
#这道题十分重要！！！！
def findNotBiggerThanMid(self, matrix, mid):
    # 以列为单位找，找到每一列最后一个<=mid的数即知道每一列有多少个数<=mid
    row, col = len(matrix), len(matrix[0])
    i, j = row - 1, 0
    count = 0
    while i >= 0 and j < col:
        if matrix[i][j] <= mid:
            count += i + 1  # 第j列有i+1个元素<=mid
            j += 1
        else:
            i -= 1  # 第j列目前的数大于mid，需要继续在当前列往上找
    return count
def kthSmallest(self, matrix, k: int) -> int:
    '''思路：一维数组找第k小:快排算法；堆排序算法
    暴力解:转为一维数组，排序可得
    既然是有序矩阵：考虑到有序性=>二分法
    这道题和n个有序数组归并或者找第k小有什么关联?????=>本质一样'''
    # 二分法，也是分治法，递归法
    row, col = len(matrix), len(matrix[0])
    left, right = matrix[0][0], matrix[row - 1][col - 1]
    while left < right:
        # 每次循环都保证第K小的数在start~end之间，当start==end，第k小的数就是start #这句话很重要!!
        mid = (left + right) // 2
        # 二维矩阵中<=mid的元素总个数
        count = self.findNotBiggerThanMid(matrix, mid)
        if count < k:
            left = mid + 1  # 第k小的数在右半部分，且不包含mid
        elif count >= k:  # 第k小的数在左半部分，可能包含mid
            right = mid
    return right

#优先队列解法




#645. 错误的集合：找到重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。[1~n]里面有个数被替为另一个数
def findErrorNums(self, nums) :
    '''思路：使用hash或者一个数组统计次数，找到次数为2和0个数'''
    n=len(nums)
    freq=[0]*n
    for i in nums:
        freq[i-1]+=1
    a=b=0
    for i in range(n):
        if freq[i]==2:
            a=i+1
        elif freq[i]==0:
            b=i+1
    return [a,b]

#通过交换数组元素，使得数组上的元素在正确的位置上。

#287. 寻找重复数:一个包含 n + 1 个整数的数组 nums，其数字都在 1 ~ n 。假设只有一个重复的整数，找出这个重复的数。
def findDuplicate(self, nums) -> int:
    '''数组查找/排序问题：只能是O(1)空间，则不能用hash/数组存储次数
    时间复杂度小于O(N*N)，则不能暴力遍历
    故：1.先排序再查找，但是会改变原来数组，不可取
    2.二分查找法=>二分法还可以用于确定一个有范围的整数（这个思路很常见）'''
    size = len(nums)
    left = 1
    right = size - 1

    while left < right:
        mid = left + (right - left) // 2

        cnt = 0
        for num in nums:
            if num <= mid:
                cnt += 1
        # 根据抽屉原理，小于等于 4 的数的个数如果严格大于 4 个，
        # 此时重复元素一定出现在 [1, 4] 区间里

        if cnt > mid:
            # 重复的元素一定出现在 [left, mid] 区间里
            right = mid
        else:
            # if 分析正确了以后，else 搜索的区间就是 if 的反面
            # [mid + 1, right]
            left = mid + 1
    return left

#667. 优美的排列：给定两个整数 n 和 k，你需要实现一个数组，这个数组包含从 1 到 n 的 n 个不同整数，同时满足以下条件：
#如果这个数组是 [a1, ..., an] ，那么数组 [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|] 中应该有且仅有 k 个不同整数；.
def constructArray(self, n: int, k: int) :
    '''思路：仅有k个不同的整数
    一开始是有序的1,2,3,4,5,6,7
    k=1 则必须保持有序；
    k=2 1,7,6,5,4,3,2
    k=3 1,7,2,3,4,5,6
    k=4 1,7,2,6,5,4,3 ==>归纳出规律为翻转 ==>所有的算法都是归纳总结'''

    res = list(range(1, n + 1))  # 刚开始有一个不同的差绝对值
    for i in range(1, k):  # 每翻转后面一次产生一个新的
        res[i:] = res[:i - 1:-1]  # 翻转
    return res

#697. 数组的度:给定一个非空且只包含非负数的整数数组 nums, 数组的度的定义是指数组里任一元素出现频数的最大值。
#你的任务是找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。
def findShortestSubArray(self, nums) -> int:
    '''思路：先找到数组的度：找到度最大的元素的首尾位置即可=>用hash保存left和right
    数组首先想到：1.是什么数据类型2.要执行什么操作：查找/排序/其他
    本题又出现最短子数组=>动态规划/状态更新等'''
    left, right, count = {}, {}, {}
    for i, x in enumerate(nums):
        if x not in left: left[x] = i
        right[x] = i
        count[x] = count.get(x, 0) + 1

    ans = len(nums)
    degree = max(count.values())
    for x in count:
        if count[x] == degree:
            ans = min(ans, right[x] - left[x] + 1)

    return ans

#766. 托普利茨矩阵:矩阵的每一方向由左上到右下的对角线上具有相同元素，那么这个矩阵是托普利茨矩阵。
def check(self, matrix, row, col, val):
    if row >= len(matrix) or col >= len(matrix[0]):
        return True
    if matrix[row][col] != val:
        return False
    return self.check(matrix, row + 1, col + 1, val)
def isToeplitzMatrix(self, matrix) -> bool:
    '''思路：只能是依次判断：从左下角开始
    思考二维矩阵的本质是连续存储：i*col+j'''
    row, col = len(matrix), len(matrix[0])
    # 判断以第一行和第一列开头的等间隔col+1位置是否相等
    for i in range(0, row):
        if self.check(matrix, i, 0, matrix[i][0]) == False:
            return False
    for j in range(0, col):
        if self.check(matrix, 0, j, matrix[0][j]) == False:
            return False
    return True

#565. 数组嵌套:S[i] 表示一个集合，集合的第一个元素是 A[i]，第二个元素是 A[A[i]]，如此嵌套下去。求最大的 S[i]。
def arrayNesting(self, nums) -> int:
    '''思路：从是s[0]开始一直找下去=>并且使用visit表示是否访问过，避免重复
    由于s[0]都是整数，如果直接在s[0]上面做标记则节省空间'''
    n = len(nums)
    visit = [0] * n
    res = 0
    for i in range(n):
        cnt = 0
        j = i
        while visit[j] == 0:
            cnt += 1
            visit[j] = 1  # 标记该位置已经被访问
            j = nums[j]
        res = max(res, cnt)
    return res


#769. 最多能完成排序的块:arr是[0, 1, ..., arr.length - 1]的一种排列，我们将这个数组分割成几个“块”，
# 并将这些块分别进行排序。之后再连接起来，使得连接的结果和按升序排序后的原数组相同。我们最多能将数组分成多少块？
def maxChunksToSorted(self, arr) -> int:
    '''思路：首先是数组问题，要求是第i块长度为n，则元素是k~k+n
    从第一个元素开始判断
    本质上：这道题和区间调度十分相似'''
    if not arr:
        return 0
    res = 0
    end = arr[0]
    for i in range(len(arr)):
        if arr[i] > end:  # 更新边界
            end = arr[i]
        if end == i:  # 组成区间边界条件
            res += 1
    return res


```

