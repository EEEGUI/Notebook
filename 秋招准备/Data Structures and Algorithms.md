# 1. Array
# 2. 链表
## 经典例题
### 链表插入排序

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        
        dummy = ListNode(-float('inf'))
        
        cur = head  # 旧链表的头
        p = dummy   # 新链表的头, 新链表维持有序
        rear = dummy    # 新链表的尾,储存最大的数
        while cur:
            if cur.val > rear.val:
                temp = cur.next
                cur.next = None
                rear.next = cur
                rear = rear.next
                cur = temp
                continue
                
            while p.next and p.next.val < cur.val:
                p = p.next
            
            # 此时p指向新链表中第一个大于当前值的结点的前一个结点
            # 在新链表中插入cur结点
            temp = cur.next
            cur.next = p.next
            p.next = cur
            cur = temp
            
            # 让p重新指向头结点
            p = dummy
        
        return dummy.next
            

```
# 3. 栈和队列
# 4. 堆
# 5. 树
## 二叉排序树
### 二叉排序树
**二叉排序树的检索**  
```python
def bt_search(btree, key):
    bt = btree:
    while bt:
        if key < bt.val:
            bt = bt.left
        elif key > bt.val:
            bt = bt.right
        else:
            return bt.val
    
    return None
```
**二叉排序树数据插入**
思路:
- 若树为空,直接新建结点
- 若树不为空:
- * 应该向左走而左结点为空,插入;
- * 应该向右走而右结点为空,插入;
- * 覆盖原有结点
```python
def bt_insert(btree, key):
    if btree is None:
        return TreeNode(key)
    bt = btree
    while True:
        if key < bt.val:
            if bt.letf is None:
                bt.left = TreeNode(key)
                return btree
            bt = bt.left
        elif key > bt.val:
            if bt.right is None:
                bt.right = Tree.Node(key)
                return btree
            bt = bt.right
        else:
            bt.val = key
            return btree
```
**二叉排序树数据删除**
思路:
- 找到待删除数值和它的父节点在树中的位置
- 若子结点的左子树为空,则直接将子节点的右子树挂到父节点,此处有3种情况
- * 父节点为空(子节点为根节点),则直接将根节点设为子节点的右子树
- * 子节点是父节点的左结点,则将子节点的右子树挂到父节点的左结点
- * 子节点是父节点的右结点,则将子节点的右子树挂到父节点的右结点
- 若子节点的左子树不为空,先找到左子树的最右结点,并将子节点的右子树挂到最右结点的右结点上,后续同样有三种情况
- * 父节点为空(子节点为根节点),则直接将根节点设为子节点的左子树
- * 子节点是父节点的左结点,则将子节点的左子树挂到父节点的左结点
- * 子节点是父节点的右结点,则将子节点的左子树挂到父节点的右结点

![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_021.png)

```python
def delete(btree, key):
    bt = btree
    father, child = None, btree
    while child and child.val != key:
        father = child
        if key < child.val:
            child = child.left
        elif key > child.val:
            child = child.right
    if child is None:
        return 
    
    # 子节点左子树为空
    if child.left is None:
        if father is None:
            btree = child.right
        elif child is father.left:
            father.left = child.right
        else:
            father.right = child.right
        return btree

    # 子节点左子树不为空
    most_right = child.left
    while most_right.right:
        most_right = most_right.right
    
    most_right.right = child.right
    if father is None:
        btree = child.left
    elif child is father.left:
        father.left = child.left
    else:
        father.right = child.left
    return btree
```

### 平衡二叉排序树

### 23树
### 红黑树
# 6. 图
# 7. 排序
## 7.1. 排序方法汇总
### 7.1.1. 冒泡排序
每一步都通过交换，把大的数后移。每一轮都有一个大的数字就位。
```python
def bubble_sort(list):
    n = len(list)
    # 每一轮选出一个最大的数值的沉入数组的尾端
    for i in range(n-1):
        flag = True
        # 在第i轮，剩余数组的长度为n-i，由于两两比较，所以比较次数为剩余数组长度再减一。
        for j in range(n-i-1):  
            # 严格大于才会调序，保证算法的稳定性
            if list[j] > list[j+1]:
                list[j], list[j+1] = list[j+1], list[j]
                # 若发生了调整，则说明当前排序仍未完成
                flag = False
        if flag:
            return list
    return list
```

链表实现
```python
def bubble_sort(head):
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        q = None    # 已有序的头结点
        while head.next != q:
            flag = False
            p = head
            while p.next != q:
                if p.val > p.next.val:
                    p.val, p.next.val = p.next.val, p.val
                    flag = True
                p = p.next
            
            q = p
            if not flag:
                break
        return head
```
最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
$O(n^2)$|$O(n^2)$|$O(n)$|$O(1)$|有|有
### 7.1.2. 插入排序
从未排序的序列中选一个数值插入到已排序的序列中
```python
def insert_sort(list):
    n = len(list)
    for i in range(n):
        temp = list[i]
        empty = i   # 取出数值，认为该位置为空
        # 空位置左侧的数字大于temp时，右移数字
        while empty > 0 and list[empty-1] > temp
                list[empty] = list[empty-1]
                # 数字后移后，空位置前移
                empty -= 1
                
        # 在空缺位置填入值
        list[empty] = temp
    return list
```
最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
$O(n^2)$|$O(n^2)$|$O(n)$|$O(1)$|有|有

### 7.1.3. 选择排序
每次从无序的序列中选择一个最小数值，放到已排序序列的末尾
```python
def select_sort(list):
    n = len(list)
    for i in range(n):
        for j in range(i, n):
            if list[j] < list[i]:
                list[j], list[i] = list[i], list[j]
    return list
```

选择排序链表实现
```python
def select_sort(head):
    if not head or not head.next:
        return head
    p = ListNode(0) # 辅助的头结点
    p.next = head   
    rear = p    # 已经排好序的尾结点

    while rear.next:
        min_node = rear.next
        q = rear.next.next
        # 找到未排序序列中最小节点
        while q:
            if q.val < min_node.val:
                min_node = q
            q = q.next
        rear.next.val, min_node.val = min_node.val, rear.next.val
        rear = rear.next
    
    return p.next

```
最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
$O(n^2)$|$O(n^2)$|$O(n^2)$|$O(1)$|无|无

### 7.1.4. 希尔排序
[希尔排序](https://zh.wikipedia.org/wiki/%E5%B8%8C%E5%B0%94%E6%8E%92%E5%BA%8F)以不同的gap进行插入排序.
```python
def shell_sort(list):
    n = len(list)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = list[i]
            empty = i
            while empty > gap and list[empty - gap] > temp:
                list[empty] = list[empty - gap]
                empty -= gap
        gap = gap // 2
    return list
```
最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
随步长而定|随步长而定|$O(n)$|$O(1)$|无|有

|步长序列|最坏时间复杂度|
|:-:|:-:|
|$n/2^i$|$O(n^2)$|
|$2^k-1$|$O(n^{3/2})$|
|$2^i3^j$|$O(nlog^2n)$|

### 7.1.5. [归并排序](https://zh.wikipedia.org/wiki/%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F)
**递归实现**
基于数据
```python
def merge(left, right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    if left:
        result += left
    if right:
        result += right
    
    return result

def merge_sort(list):
    if len(list) <= 1:
        return list
    
    mid = len(list) // 2
    left = list[:mid]
    right = list[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)
```
基于链表:
时间复杂度$O(nlogn)$, 空间复杂度不计递归栈为$O(1)$
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def merge(h1, h2):
        # 两个有序链表的合并
            p1 = ListNode(0)
            p1.next = h1
            
            record = p1
            
            while h1 and h2:
                if h1.val <= h2.val:
                    p1 = p1.next
                    h1 = h1.next
                else:
                    temp = h2.next
                    h2.next = p1.next
                    p1.next = h2
                    h2 = temp
                    
                    p1 = p1.next
                    
            if h2:
                p1.next = h2
                
            return record.next
        
        def merge_sort(head):
            if not head or not head.next:
                return head
            # 快慢指针,找到中间结点
            slow = head
            fast = head.next
            head1 = head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            
            head2 = slow.next
            slow.next = None
            
            h1 = merge_sort(head1)
            h2 = merge_sort(head2)
            return merge(h1, h2)
        
        return merge_sort(head)
```
**迭代实现**
```python
def merge(lfrom, lto, low, mid, high):
    i = low
    j = mid
    k = low
    while i < mid and j < high:
        if lfrom[i] <= lfrom[j]:
            lto[k] = lfrom[i]
            i += 1
        else:
            lto[k] = lfrom[j]
            j += 1
        k += 1

    while i < mid:
        lto[k] = lfrom[i]
        i += 1
        k += 1

    while j < high:
        lto[k] = lfrom[j]
        j += 1
        k += 1
    return lfrom, lto

def merge_pass(lfrom, llen, slen):
    i = 0
    lto = [None] * len(lfrom)
    while i + 2 * slen < llen:
        lfrom, lto = merge(lfrom, lto, i, i+slen, i+2*slen)
        i += 2 * slen
    # 剩余两段,第二段长度小鱼slen
    if i + slen < llen:
        lfrom, lto = merge(lfrom, lto, i, i+slen, llen)
    # 只剩余一段,直接复制
    else:
        while i < llen:
            lto[i] = lfrom[i]
            i+=1
    return lto

def merge_sort(lst):
    slen = 1
    llen = len(lst)
    while slen < llen:
        lst = merge_pass(lst, llen, slen)
        slen *= 2
    return lst
```
每经过一遍归并,有序子序列的长度将变成$2^k$,因此归并的遍数不会超过$log_2n+1$,而每遍归并时,比较次数为$O(n)$,因此时间复杂度为$O(nlogn)$  
最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
$O(nlogn)$|$O(nlogn)$|$O(nlogn)$|$O(n)$|有|无

### 7.1.6. 快速排序
```python
def quick_rec(lst, b, e):
    if e <= b:
        return lst
    temp = lst[b]
    i = b
    j = e
    while i < j:
        # 严格加上i<j,否则会侵入已经分好的区域
        while i < j and lst[j] > temp:
            j -= 1
        if i < j:
            lst[i] = lst[j]
            i += 1
        while i < j and lst[i] < temp:
            i += 1
        if i < j:
            lst[j] = lst[i]
            j -= 1
    lst[i] = temp
    lst = quick_rec(lst, b, i-1)
    lst = quick_rec(lst, i+1, e)
    return lst

def quick_sort(lst):
    return quick_rec(lst, 0, len(lst)-1)
```

快速排序链表实现
```python
def quick_sort(head):

    def partition(head, tail):
        # 包含head, 不包含tail
        key = head.val
        mid = head
        p = head.next
        while p!=tail:
            if p.val < key:
                mid = mid.next
                mid.val, p.val = p.val, mid.val
            p = p.next
        mid.val, head.val = head.val, mid.val
        return mid

    def q_sort(head, tail):
        if head != tail and head.next != tail:
            mid = partition(head, tail)
            q_sort(head, mid)
            q_sort(mid.next, tail)

    if not head or not head.next:
        return head
    q_sort(head, None)
    return head

```

最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
$O(n^2)$|$O(nlogn)$|$O(nlogn)$|$O(1)$|无|无

### 7.1.7. [堆排序](https://zh.wikipedia.org/wiki/%E5%A0%86%E6%8E%92%E5%BA%8F)
构建一个最大堆,每次从最大堆中去除最大元素,放到数组的末尾,完成排序.
```python
def heap_sort(lst):
    def sift_down(start, end):
        """
        从上到下调整节点,若父节点的值小于子节点,交换位置.
        """
        root = start
        child = 2 * root + 1
        while child < end:
            if child+1 < end and lst[child] < lst[child+1]:
                child += 1
            if lst[child] > lst[root]:
                lst[root], lst[child] = lst[child], lst[root]
                root, child = child, 2*child + 1
            else:
                break
    n = len(lst)
    # 建最大堆
    # len(lst)//2 -1 为最后一个树节点,其子节点都为叶节点
    for start in range(n//2-1, -1, -1):
        sift_down(start, n-1)
    
    # 从最大堆中取值
    for i in range(n-1, 0, -1):
        lst[i], lst[0] = lst[0], lst[i]
        sift_down(0, i-1)
    return lst
```
最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
$O(nlogn)$|$O(nlogn)$|$O(nlogn)$|$O(1)$|无|无

### 7.1.8. 计数排序
当待排序序列中,不同数字的个数较少时(例如0-k之间的整数),计数排序是最快的方法.
```python
def count_sort(lst):
    min_ = min(lst)
    max_ = max(lst)

    count = [0] * (max_ - min_ + 1)

    # 统计每一个数字出现的次数
    for num in lst:
        count[num-min_] += 1
    
    index = 0
    # 根据统计的结果,将数值回填到原来的list中
    for i, times in enumerate(count):
        for j in range(times):
            lst[index] = i + min_
            index += 1
    return lst
```

最坏时间复杂度|平均时间复杂度|最好时间复杂度|空间复杂度|稳定性|适应性
:-:|:-:|:-:|:-:|:-:|:-:|
$O(n+k)$|$O(n+k)$|$O(n+k)$|$O(n+k)$|无|无
## 7.2. 经典例题
### 7.2.1. 求最小的K个数
We have a list of points on the plane.  Find the K closest points to the origin (0, 0).

(Here, the distance between two points on a plane is the Euclidean distance.)

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)  
采用类似于快排分区的方法,时间复杂度:$O(N)$, 空间复杂度$O(N)$
采用最大堆的方法,时间复杂度:$O(NlogK)$, 空间复杂度$O(N)$

```python
class Solution(object):
    def kClosest(self, points, K):
        dist = lambda i: points[i][0]**2 + points[i][1]**2

        def sort(i, j, K):
            # Partially sorts A[i:j+1] so the first K elements are
            # the smallest K elements.
            if i >= j: return

            # Put random element as A[i] - this is the pivot
            k = random.randint(i, j)
            points[i], points[k] = points[k], points[i]

            mid = partition(i, j)
            if K < mid - i + 1:
                sort(i, mid - 1, K)
            elif K > mid - i + 1:
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i, j):
            # Partition by pivot A[i], returning an index mid
            # such that A[i] <= A[mid] <= A[j] for i < mid < j.
            oi = i
            pivot = dist(i)
            i += 1

            while True:
                while i < j and dist(i) < pivot:
                    i += 1
                while i <= j and dist(j) >= pivot:
                    j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]

            points[oi], points[j] = points[j], points[oi]
            return j

        sort(0, len(points) - 1, K)
        return points[:K]
```
### 字符统计并排序
Given a string S, check if the letters can be rearranged so that two characters that are adjacent to each other are not the same.

If possible, output any possible result.  If not possible, return the empty string.
```
Example 1:

Input: S = "aab"
Output: "aba"
Example 2:

Input: S = "aaab"
Output: ""
```
Note:

S will consist of lowercase letters and have length in range [1, 500].

```python
class Solution:
    def reorganizeString(self, S: str) -> str:
        n = len(S)
        A = []
        
        for c, x in sorted([S.count(x),x] for x in set(S)):
            if c > (n+1)/2:
                return ''
            A.extend(c * x)
        
        ans = [None] * n
        ans[::2], ans[1::2] = A[n//2:], A[:n//2]
        
        return ''.join(ans)
```
### 相同数字不相邻
1054 Distant Barcodes  
In a warehouse, there is a row of barcodes, where the i-th barcode is barcodes[i].

Rearrange the barcodes so that no two adjacent barcodes are equal.  You may return any answer, and it is guaranteed an answer exists.

 
```
Example 1:

Input: [1,1,1,2,2,2]
Output: [2,1,2,1,2,1]
Example 2:

Input: [1,1,1,1,2,2,3,3]
Output: [1,3,1,3,2,1,2,1]
```
此处没有采用新生成数组A,直接在原数组中,利用数字出现个数进行排序,减少运行时间.
```python
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        n = len(barcodes)
        if n <= 1:
            return barcodes
        count = {}
        
        for each in barcodes:
            if each in count:
                count[each] += 1
            else:
                count[each] = 1
                
        barcodes.sort(key=lambda x: (count[x], x))
        ans = [None] * n
        ans[::2] = barcodes[n//2:]
        ans[1::2] = barcodes[:n//2]
        return ans
```

### Contains Duplicate III
Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.
```
Example 1:

Input: nums = [1,2,3,1], k = 3, t = 0
Output: true
Example 2:

Input: nums = [1,0,1,1], k = 1, t = 2
Output: true
Example 3:

Input: nums = [1,5,9,1,5,9], k = 2, t = 3
Output: false
```
```python
class Solution:

    def containsNearbyAlmostDuplicate(self, nums, k, t):
        if t < 0: return False
        n = len(nums)
        d = {}
        w = t + 1
        for i in range(n):
            m = nums[i] // w
            if m in d:
                return True
            if m - 1 in d and abs(nums[i] - d[m - 1]) < w:
                return True
            if m + 1 in d and abs(nums[i] - d[m + 1]) < w:
                return True
            d[m] = nums[i]
            if i >= k: del d[nums[i - k] // w]
        return False

```
# 8. 搜索
# 9. 动态规划
# 10. 回溯法