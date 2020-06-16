#### 排序算法总结

```python
def select_sort(array):#二个for
    #思想：依次选择未排序中最小的元素与未排序的首个元素交换
    for i in range(0,len(array)):
        min_index=i
        #找到未排序中最小的元素
        for j in range(i+1,len(array)):
            if array[min_index]>array[j]:
                min_index=j
        array[min_index],array[i]=array[i],array[min_index]
    return array


def insert_sort(array): #一个for 一个while
    #依次将未排序的首个元素插入已排序的合适位置(从已排序中由后往前扫描)
    #对部分有序非常有效，也适合小规模数组
    for i in range(1,len(array)):
        cur=array[i]
        j=i-1
        while j>=0 and cur<array[j]:
            array[j+1]=array[j]
            j-=1
        array[j+1]=cur
    return array


def bubble_sort(array): #二个for循环
    for i in range(0,len(array)):
        for j in range(0,len(array)-i-1):
            if array[j]>array[j+1]:
                array[j+1],array[j]=array[j],array[j+1]
    return array


def shell_sort(array): #while for for while[最后二个是插入排序]
    #对插入排序的改进,交换不相邻的元素对数组局部排序，并最终使用插入排序=>可用于中等大小数组和随机数组
    #思想：使数组中任意间隔为h的元素都是有序的=>还有更好的写法，但是没怎么看懂
    n=len(array)
    gap=n//2 #组数
    while gap>0:#逐渐减小gap，直到为1
        for i in range(0,gap):#对每一组进行插入排序,每一组开头元素为i
            for j in range(i+gap,n,gap):#对未排序部分插入
                temp=array[j]
                k=j-gap #排好序部分的最后一个元素，即未排序的前一个元素
                while k>=0 and temp<array[k]:
                    array[k+gap]=array[k]  #元素后移
                    k-=gap
                array[k+gap]=temp

        gap//=2
    return array


#参考https://cloud.tencent.com/developer/article/1574734
def merge(s1,s2,s):
    """将两个列表是s1，s2按顺序融合为一个列表s,s为原列表"""
    # j和i就相当于两个指向的位置，i指s1，j指s2
    i=j=0
    while i+j<len(s):
        # j==len(s2)时说明s2走完了，或者s1没走完并且s1中该位置是最小的
        if j==len(s2) or (i<len(s1) and s1[i]<s2[j]):
            s[i+j]=s1[i]
            i+=1
        else:
            s[i+j]=s2[j]
            j+=1
def merge_sort(s):
    n=len(s)
    if n<=1:
        return
    #拆分
    mid=n//2
    s1=s[0:mid]
    s2=s[mid:n]

    #子序列调用递归排序
    merge_sort(s1)
    merge_sort(s2)
    #合并
    merge(s1,s2,s)


#https://blog.csdn.net/weixin_43250623/article/details/88931925
def patition(array,left,right):
    pivot=array[left]
    i=left+1
    j=right
    while i<=j:
        while i<=j and array[i]<=pivot:
            i+=1
        while i<=j and array[j]>=pivot:
            j-=1
        if i<=j:
            array[i],array[j]=array[j],array[i]
            i,j=i+1,j-1
    array[left],array[j]=array[j],array[left]
    return j
def quick_sort(array,left,right):
    if left<right:
        mid=patition(array,left,right)
        quick_sort(array,left,mid-1)
        quick_sort(array,mid+1,right)
    return array


def count_sort(arr):
    #计数排序适用于arr中数据在一定的范围中
    #第一步求最值
    max_num=max(arr)
    min_num=min(arr)

    #第二步创建统计数组
    count_arr=[0]*(max_num-min_num+1) #创建统计数组
    for i in arr:
        count_arr[i-min_num]+=1

    arr.clear()
    for index,i in enumerate(count_arr):
        while i:
            arr.append(index+min_num)
            i-=1

    return arr


def bucket_sort(s):
    """桶排序"""
    min_num = min(s)
    max_num = max(s)
    # 每个桶的大小
    bucket_range = (max_num-min_num) / len(s)
    # 桶数组
    count_list = [ [] for i in range(len(s) + 1)]
    # 向桶数组填数
    for i in s:
        count_list[int((i-min_num)//bucket_range)].append(i)
    s.clear()
    # 回填，这里桶内部排序直接调用了sorted
    for i in count_list:
        for j in sorted(i):
            s.append(j)

    return s


if __name__=='__main__':

    print(bucket_sort([55,52,53,54,56,68,69,67,67,68,57,59]))

```

