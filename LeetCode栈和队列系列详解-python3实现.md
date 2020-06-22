#### #栈和队列：单调栈法解决next greater element

```python

#栈和队列
#20. 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
def isValid(self, s: str) -> bool:
    # 思路：凡是遇到左边全部入栈，一旦遇到右边则匹配栈顶并弹出
    n = len(s)
    stack = []
    left = ['(', '[', '{']
    right = [')', ']', '}']
    for i in range(0, n):
        if s[i] in left:
            stack.append(s[i])
        elif (s[i] in right) and stack:
            if left.index(stack[-1]) == right.index(s[i]):
                stack.pop(-1)
            else:
                return False
        else:
            return False
    return stack == []

#739. 每日温度：每日气温列表，生成一个列表其对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。
# 如果气温在这之后都不会升高，请在该位置用 0 来代替。
#也就是找下一个比当前大的数字的位置
def dailyTemperatures(self, T):
    '''思路1：遍历法
    思路二：单调栈法解next greater element 问题'''
    stack = []
    res = [0] * len(T)
    for i in range(len(T) - 1, -1, -1):
        while stack and T[i] >= T[stack[-1]]:
            stack.pop(-1)
        if stack:
            res[i] = stack[-1] - i
        stack.append(i)
    return res


#503. 下一个更大元素：数组是循环数组，并且最后要求的不是距离而是下一个元素。
def nextGreaterElements(self, nums):
    if not nums:
        return []
    length = len(nums)

    nums = nums + nums
    result = [-1] * length
    stack = [0]

    for i in range(1, len(nums)):
        while (stack and nums[stack[-1]] < nums[i]):
            target = stack.pop()
            result[target % length] = nums[i]
        stack.append(i % length)
    return result

```

