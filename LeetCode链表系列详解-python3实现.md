#### #总结链表问题：递归+迭代+双指针

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

        
#160. 相交链表：找到两个单链表相交的起始节点。
def getIntersectionNode(self, headA,headB):
    '''思路：链表首先想到双指针和递归
    本题思路：'''
    p1 = headA
    p2 = headB
    while p1 != p2:
        p1 = p1.next if p1.next else headB
        p2 = p2.next if p2.next else headA
    return p1

#206. 反转链表
#递归法
def reverseList(self, head) :
    #递归法：
    #递归的三个步骤：
    #1.明确函数定义以及返回值，切记跳入细节
    #2.递归结束条件和当前输入应该做什么
    #3.调用递归框架
    if not head or not head.next:
        return head
    new=self.reverseList(head.next)
    #该函数始终没有对head节点做任何处理，因此需要加上：
    nxt=head.next
    head.next=None
    nxt.next=head

    return new
#迭代法
def reverseList(self, head):
    '''链表问题：首先想到双指针和递归'''
    pre = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    return pre

#21. 合并两个有序链表
#迭代写法
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    '''思路：和归并排序的merge函数十分相似'''
    p1, p2 = l1, l2
    head = new = ListNode(0)
    while p1 or p2:
        if not p1 or (p2 and p2.val <= p1.val):
            new.next = ListNode(p2.val)
            p2 = p2.next
        else:
            new.next = ListNode(p1.val)
            p1 = p1.next
        new = new.next
    new.next = None
    return head.next
#递归写法
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    '''思路：和归并排序的merge函数十分相似'''
    # 递归写法 牢记3个步骤
    if not l1:
        return l2
    if not l2:
        return l1

    if l1.val <= l2.val:
        l1.next = self.mergeTwoLists(l1.next, l2)
    else:
        l2.next = self.mergeTwoLists(l1, l2.next)
    return l1 if l1.val <= l2.val else l2

#83. 删除排序链表中的重复元素 ： 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
#迭代法
def deleteDuplicates(self, head: ListNode) -> ListNode:
    p=head
    while p and p.next:
        if p.val==p.next.val:
            p.next=p.next.next
        else:
            p=p.next

    return head

#递归法
def deleteDuplicates(self, head: ListNode) -> ListNode:
    '''思路：链表问题：递归+迭代+双指针'''
    # 递归法:牢记递归3步骤
    if not head or not head.next:
        return head

    head.next = self.deleteDuplicates(head.next)  # 调用递归框架
    # 明确返回值类型以约束
    return head.next if head.val == head.next.val else head

#19. 删除链表的倒数第N个节点
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    '''要想删除首先要找到他的前驱节点：
    链表首先想到双指针+递归+迭代
    找到倒数节点可以使用双指针'''
    p1 = p2 = head
    while n > 0:
        p1 = p1.next
        n -= 1
    if not p1:
        return head.next
    while p1.next:
        p1 = p1.next
        p2 = p2.next
    p2.next = p2.next.next
    return head

#24. 两两交换链表中的节点：你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。


#445. 两数相加:两个非空链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。
def isPalindrome(self, head: ListNode) -> bool:
    '''链表首先想到 双指针+递归+迭代
    判断回文链表：1：后半部分反转+双指针  2：使用栈'''
    if not head or not head.next:
        return True
    fast = slow = head
    stack = []
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    while slow:
        stack.append(slow.val)
        slow = slow.next
    while stack:
        if stack.pop(-1) != head.val:
            return False
        head = head.next
    return True


#725. 分隔链表：将链表分隔为 k 个连续的部分，每部分的长度应该尽可能的相等: 任意两部分的长度差距不能超过 1；
#并且排在前面的部分的长度应该大于或等于后面的长度
```

