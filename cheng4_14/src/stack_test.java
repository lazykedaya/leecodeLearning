import com.sun.security.auth.NTNumericCredential;

import javax.swing.*;
import javax.swing.text.AbstractDocument;
import java.util.*;

public class stack_test {
    public static void main(String[] args) {
        ListNode n1 = new ListNode(1);
        n1.next = new ListNode(4);
        n1.next.next = new ListNode(3);
        n1.next.next.next = new ListNode(2);
        n1.next.next.next.next = new ListNode(5);
        ListNode n2 = new ListNode(1);
        n2.next = new ListNode(3);
        n2.next.next = new ListNode(4);
        TreeNode root = new TreeNode(2);
        root.left = new TreeNode(1);
        root.right = new TreeNode(3);
        BSTIterator bst = new BSTIterator(root);
        System.out.println(bst.hasNext());


    }
    //二叉树的最近公共祖先
    public static TreeNode ans2=new TreeNode();
    public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null)return null;
        dfs2(root, p, q);
        return ans2;
    }

    private static boolean dfs2(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null)return false;
        boolean L = dfs2(root.left, p, q);
        boolean R = dfs2(root.right, p, q);
        if((L&&R)||(root.val==p.val||root.val==q.val)&&(L||R)){
            ans2=root;
        }
        return L||R||p.val==root.val||q.val==root.val;
    }

    //二叉搜索树的最小绝对差//第K小的元素
    public static int getMinimumDifference(TreeNode root,int k) {
     //二叉搜索树的中序遍历是有序递增的，因此使用中序遍历
        /*LinkedList<TreeNode> stack = new LinkedList<>();
        TreeNode pNode = root;
        int pre=Integer.MAX_VALUE;
        int ans=Integer.MAX_VALUE;
        while (pNode != null || !stack.isEmpty()) {
            if (pNode != null) {
                stack.push(pNode);
                pNode = pNode.left;
            } else { //pNode == null && !stack.isEmpty()
                TreeNode node = stack.pop();
                ans=Math.min(ans,Math.abs(node.val-pre));
                pre=node.val;
                pNode = node.right;
            }
        }
        return ans;*/
        LinkedList<TreeNode> stack = new LinkedList<>();
        TreeNode pNode = root;
        int ans=0;
        while (pNode != null || !stack.isEmpty()) {
            if (pNode != null) {
                stack.push(pNode);
                pNode = pNode.left;
            } else { //pNode == null && !stack.isEmpty()
                TreeNode node = stack.pop();
                if(k==1){
                    ans=node.val;
                    break;
                }else k--;
                pNode = node.right;
            }
        }
        return ans;
    }

    //二叉树的层序遍历以及 二叉树的锯齿形层序遍历
    public static List<List<Integer>> ans = new ArrayList<>();
    public static List<List<Integer>> levelOrder(TreeNode root) {
        //和后面几题层序遍历的没啥区别
        /*ArrayList<List<Integer>> ans = new ArrayList<>();
        if(root==null)return ans;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size=queue.size();
            ArrayList<Integer> temp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if(node.left!=null)queue.offer(node.left);
                if(node.right!=null)queue.offer(node.right);
                temp.add(node.val);
            }
            ans.add(temp);
        }
        return ans;*/
        //换成dfs
        f1(root, 0);
        return ans;
    }
    private static void f1(TreeNode root, int depth) {
        if (root == null) return;
        depth++;
        if (ans.size() < depth) {
            List<Integer> list = new LinkedList<>();
            ans.add(list);
        }
        if (depth % 2 == 1) {
            ans.get(depth - 1).add(root.val);
        }
        else ans.get(depth-1).add(0,root.val);
        f1(root.left, depth);
        f1(root.right, depth);
    }

    //二叉树的右视图
    public static List<Integer> rightSideView(TreeNode root) {
        ArrayList<Integer> ans = new ArrayList<>();
        if (root == null) return ans;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
                if (i == size - 1) ans.add(node.val);
            }
        }
        return ans;
    }

    // 二叉树的层平均值
    public static List<Double> averageOfLevels(TreeNode root) {
        ArrayList<Double> ans = new ArrayList<>();
        if (root == null) return ans;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            double sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
                sum += node.val;
            }
            ans.add(sum / size);
        }
        return ans;
    }

    //求根节点到叶节点数字之和
    public static int sumNumbers(TreeNode root) {
        if (root == null) return 0;
        //深度优先
        //return dfs(root,0);
        //广度优先
        int sum = 0;
        LinkedList<TreeNode> queue = new LinkedList<>();
        LinkedList<Integer> sumQueue = new LinkedList<>();
        queue.offer(root);
        sumQueue.offer(root.val);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            int num = sumQueue.poll();
            TreeNode left = node.left, right = node.right;
            if (left == null && right == null) sum += num;
            else {
                if (left != null) {
                    queue.offer(left);
                    sumQueue.offer(num * 10 + left.val);
                }
                if (right != null) {
                    queue.offer(right);
                    sumQueue.offer(num * 10 + right.val);
                }

            }
        }
        return sum;

    }

    private static int dfs(TreeNode root, int preSum) {
        if (root == null) return 0;
        int sum = preSum * 10 + root.val;
        if (root.left == null && root.right == null) {
            return sum;
        } else return dfs(root.left, sum) + dfs(root.right, sum);
    }

    //二叉树展开为链表
    public static void flatten(TreeNode root) {
        //用前序遍历做的
        /*if(root==null)return;
        ArrayList<TreeNode> lis = new ArrayList<>();
        preOledr(lis,root);
        int n=lis.size();
        for (int i = 1; i < n; i++) {
            TreeNode prev=lis.get(i-1),curr=lis.get(i);
            prev.left=null;
            prev.right=curr;
        }*/
        //不使用,定义curr，next，predecessor
        TreeNode curr = root;
        while (curr != null) {
            if (curr.left != null) {
                TreeNode next = curr.left;
                TreeNode pre = next;
                while (pre.right != null) {
                    pre = pre.right;
                }
                pre.right = curr.right;
                curr.left = null;
                curr.right = next;
            }
            curr = curr.right;
        }
    }

    private static void preOledr(List<TreeNode> list, TreeNode root) {
        if (root != null) {
            list.add(root);
            preOledr(list, root.left);
            preOledr(list, root.right);
        }
    }

    //填充每个节点的下一个右侧节点指针 II
    public static Tnode connect(Tnode root) {
        if (root == null) return null;
        //层次遍历
        LinkedList<Tnode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int n = queue.size();
            Tnode last = null;
            for (int i = 1; i <= n; i++) {
                Tnode f = queue.poll();
                if (f.left != null) queue.offer(f.left);
                if (f.right != null) queue.offer(f.right);
                if (i != 1) last.next = f;
                last = f;
            }

        }
        return root;

    }

    //完全二叉树的节点个数
    public static int countNodes(TreeNode root) {
        //时间复杂度可以优化
       /*if(root==null)return 0;
       int left=countNodes(root.left);
       int right=countNodes(root.right);
       return left+right+1;*/
        //利用完全二叉树的性质
        if (root == null) return 0;
        TreeNode node = root;
        int lDepth = 0, rDepth = 0;
        while (node.left != null) {
            lDepth++;
            node = node.left;
        }
        node = root;
        while (node.right != null) {
            rDepth++;
            node = node.right;
        }
        if (lDepth == rDepth) return 2 * (lDepth + 1) - 1;
        else return countNodes(root.left) + countNodes(root.right) + 1;
    }

    //路径总和
    public static boolean hasPathSum(TreeNode root, int targetSum) {
        //这里使得原树被数据污染了
        /*if(root==null)return false;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean flag=false;
        while (!queue.isEmpty()){
            TreeNode node=queue.poll();
            int a = node.val;
            if(node.left!=null){
                node.left.val+=a;
                queue.offer(node.left);
            }
            if(node.right!=null){
                node.right.val+=a;
                queue.offer(node.right);
            }
            if(node.left==null&&node.right==null&&a==targetSum)flag=true;
        }
        return flag;*/
        //可以使用递归避免
        if (root == null) return false;
        targetSum -= root.val;
        if (root.left == null && root.right == null) return targetSum == 0;
        if (root.left != null) {
            if (hasPathSum(root.left, targetSum)) return true;
        }
        if (root.right != null) {
            if (hasPathSum(root.right, targetSum)) return true;
        }
        return false;
    }

    //从中序与后序遍历序列构造二叉树
    public static HashMap<Integer, Integer> indexMap2;

    public static TreeNode buildTree2(int[] inorder, int[] postorder) {
        TreeNode ans = new TreeNode();
        int n = inorder.length;
        indexMap2 = new HashMap<>();
        //将中序数组放进map中，键为数组内值，值为数组中index。
        for (int i = 0; i < n; i++) {
            indexMap2.put(inorder[i], i);
        }
        //后序遍历最后面的为根节点
        return myBuildTree2(inorder, postorder, 0, n - 1, 0, n - 1);
    }

    public static TreeNode myBuildTree2(int[] inorder, int[] postorder, int IL, int IR, int PL, int PR) {
        if (PL > PR) return null;
        //后序遍历最后一个就是根节点
        TreeNode root = new TreeNode(postorder[PR]);
        //到中序遍历数组中找到根节点
        int inOrderRoot = indexMap2.get(postorder[PR]);
        //左子树的大小
        int sizeRightSubTree = IR - inOrderRoot;
        // 递归地构造右子树，并连接到根节点
        root.right = myBuildTree2(inorder, postorder, inOrderRoot + 1, IR, PR - sizeRightSubTree, PR - 1);
        // 递归地构造左子树，并连接到根节点
        root.left = myBuildTree2(inorder, postorder, IL, inOrderRoot - 1, PL, PR - sizeRightSubTree - 1);
        return root;
    }

    //基本计算器
    //数字不进栈，栈记录括号前的符号，遇到左括号进栈sign，遇到右括号出去一个sign
    public static int calculate(String s) {
        LinkedList<Integer> queue = new LinkedList<>();
        queue.push(1);
        int sign = 1;
        int res = 0;
        int n = s.length();
        int i = 0;
        while (i < n) {
            char c = s.charAt(i);
            if (c == ' ') i++;
            else if (c == '+') {
                sign = queue.peek();
                i++;
            } else if (c == '-') {
                sign = -queue.peek();
                i++;
            } else if (c == '(') {
                queue.push(sign);
                i++;
            } else if (c == ')') {
                queue.pop();
                i++;
            } else {
                long num = 0;
                while (i < n && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + s.charAt(i) - '0';
                    i++;
                }
                res += sign * num;
            }
        }
        return res;
    }

    //分隔链表
    public static ListNode partition(ListNode head, int x) {
        //维护两个链表，一个是小于的，一个是大于x的，最后把它们拼接起来
        if (head == null || head.next == null) return head;
        ListNode sFirst = new ListNode(0);
        ListNode sTemp = sFirst;
        ListNode sBig = new ListNode(0);
        ListNode bTemp = sBig;
        while (head != null) {
            if (head.val >= x) {
                bTemp.next = new ListNode(head.val);
                bTemp = bTemp.next;
            } else {
                sTemp.next = new ListNode(head.val);
                sTemp = sTemp.next;
            }
            head = head.next;
        }
        sTemp.next = sBig.next;
        return sFirst.next;
    }

    //K 个一组翻转链表
    public static ListNode reverseKGroup(ListNode head, int k) {
        //利用栈来实现判断与翻转
        if (head == null || head.next == null) return head;
        ListNode ans = new ListNode(0, head);
        ListNode temp = ans;
        while (head != null) {
            int count = 0;
            ListNode start = head;
            Stack<ListNode> st = new Stack<>();
            for (int i = 0; i < k; i++) {
                if (head != null) {
                    st.push(new ListNode(head.val));
                    head = head.next;
                } else break;
                count++;
            }
            if (count == k) {
                while (!st.isEmpty()) {
                    temp.next = st.pop();
                    temp = temp.next;
                }
            } else {
                temp.next = start;
                break;
            }
        }
        return ans.next;
    }

    //从前序与中序遍历序列构造二叉树
    public static Map<Integer, Integer> indexMap;

    public static TreeNode buildTree(int[] preorder, int[] inorder) {
        //前序可以先看见根节点，中序确定左节点
        TreeNode ans = new TreeNode();
        int n = preorder.length;
        //构造哈希表，定位根节点
        indexMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return myBudildTree(preorder, inorder, 0, n - 1, 0, n - 1);

    }

    public static TreeNode myBudildTree(int[] preorder, int[] inorder, int preorderLeft, int preorderRight, int inorderLeft, int inorderRight) {
        if (preorderLeft > preorderRight) {
            return null;
        }
        //前序遍历第一个就是根节点
        int inOrderRoot = indexMap.get(preorder[preorderLeft]);
        TreeNode root = new TreeNode(preorder[preorderLeft]);
        //得到左子树的节点数目
        int sizeLeftSubtree = inOrderRoot - inorderLeft;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中从左边界+1 开始的 size_left_subtree个元素就对应了中序遍历中从左边界开始到根节点定位-1的元素
        root.left = myBudildTree(preorder, inorder, preorderLeft + 1, preorderLeft + sizeLeftSubtree, inorderLeft, inOrderRoot - 1);
        root.right = myBudildTree(preorder, inorder, preorderLeft + sizeLeftSubtree + 1, preorderRight, inOrderRoot + 1, inorderRight);
        return root;
    }

    //旋转链表
    public static ListNode rotateRight(ListNode head, int k) {
        //先看大小，取余数，再把后k个转移到前面来
        if (k == 0 || head == null || head.next == null) return head;
        int n = 1;
        ListNode ans = head;
        ListNode mid = head;
        while (head.next != null) {
            head = head.next;
            n++;
        }
        //保留最后一个节点，后面用来连接到开头
        ListNode end = head;
        k = k % n;
        if (k == 0) return ans;
        for (int i = n; i > k + 1; i--) {
            ans = ans.next;
        }
        //保留截断位置节点作为初始节点
        ListNode pre = ans.next;
        //截断倒数k个
        ans.next = null;
        //最后k个连接到开头
        end.next = mid;
        return pre;
    }

    //删除排序链表中的重复元素 II
    public static ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
        ListNode ans = new ListNode(0, head);
        ListNode temp = ans;
        while (temp.next != null && temp.next.next != null) {
            if (temp.next.val == temp.next.next.val) {
                int x = temp.next.val;
                while (temp.next != null && temp.next.val == x) {
                    temp.next = temp.next.next;
                }
            } else temp = temp.next;
        }
        return ans.next;
    }

    //删除链表的倒数第 N 个结点
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        //先搞清多长，然后正向来判断
        ListNode ans = new ListNode();
        ans = head;
        int size = 0;
        while (head != null) {
            head = head.next;
            size++;
        }
        int k = size - n;
        if (k == 0 && ans != null) return ans.next;
        else if (k < 0) return ans;
        ListNode temp = new ListNode();
        temp = ans;
        for (int i = k - 1; i > 0; i--) {
            if (temp != null) temp = temp.next;
        }
        if (temp != null) temp.next = temp.next.next;
        return ans;
    }

    //对称二叉树
    public static boolean isSymmetric(TreeNode root) {
        return check1(root, root);
    }

    private static boolean check1(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        return p.val == q.val && check1(p.left, q.right) && check1(p.right, q.left);
    }

    //翻转二叉树
    public static TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    //相同的树
    public static boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        else if (p == null || q == null) return false;
        else if (p.val != q.val) return false;
        else return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    //二叉树的最大深度
    public static int maxDepth(TreeNode root) {
        //二叉树遍历
        if (root == null) return 0;
        else {
            int leftHeight = maxDepth(root.left);
            int rightHeight = maxDepth(root.right);
            return Math.max(rightHeight, leftHeight) + 1;
        }

    }

    //二叉树遍历
    public static void bianLi(TreeNode root) {
        if (root != null) {
            //sout(root.val)前序遍历
            bianLi(root.left);
            //sout(root.val)中序遍历
            bianLi(root.right);
            //sout(root.val)后序遍历
        }
    }

    //层次遍历
    public static int cengCi(TreeNode root) {
        if (root == null) return 0;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        //先在队列中加入根结点。之后对于任意一个结点来说，在其出队列的时候访问之。同时如果左孩子和右孩子有不为空的，入队列
        int num = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size > 0) {
                //poll方法返回并删除头部元素
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.left);
                }
                size--;
            }
            num++;
        }
        return num;

    }

    //*****反转链表 II
    public static ListNode reverseBetween(ListNode head, int left, int right) {
        /*先将待反转的区域反转；
         第2步：把pre的next指针指向反转以后的链表头节点，把反转以后的链表的尾节点的next指针指向suc*/
        /*ListNode cyber=new ListNode(-1);
        cyber.next=head;
        ListNode pre=cyber;
        for (int i = 0; i < left-1; i++) {
            pre=pre.next;
        }
        ListNode rightNode=pre;
        for (int i = left; i <= right; i++) {
            rightNode=rightNode.next;
        }
        ListNode leftNode=pre.next;
        ListNode curr=rightNode.next;

        pre.next=null;
        rightNode.next=null;
        //反转
        reverseList(leftNode);
        pre.next=rightNode;
        leftNode.next=curr;
        return cyber.next;*/
        ListNode ans = new ListNode(-1);
        ans.next = head;
        ListNode pre = ans;
        //找到left-1节点
        for (int i = 1; i < left; i++) {
            pre = pre.next;
        }
        //pre永远指向cur，cur指向next，next指向pre
        ListNode cur = pre.next;
        ListNode next;
        for (int i = 0; i < right - left; i++) {
            next = cur.next;
            cur.next = next.next;
            next.next = pre.next;
            pre.next = next;
        }
        return ans.next;

    }

    private static void reverseList(ListNode leftNode) {
        ListNode ppre = null;
        ListNode cur = leftNode;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = ppre;
            ppre = cur;
            cur = next;
        }
    }

    public static HashMap<Node, Node> map = new HashMap<>();

    //随机链表的复制
    public static Node copyRandomList(Node head) {
        if (head == null) return null;
        while (!map.containsKey(head)) {
            Node node = new Node(head.val);
            map.put(head, node);
            node.next = copyRandomList(head.next);
            node.random = copyRandomList(head.random);
        }
        return map.get(head);
    }

    //合并两个有序链表
    public static ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode ans = new ListNode();
        ListNode temp = ans;
        while (list1 != null || list2 != null) {
            if (list1 != null && list2 != null) {
                if (list1.val > list2.val) {
                    temp.next = new ListNode(list2.val);
                    temp = temp.next;
                    list2 = list2.next;
                } else if (list1.val < list2.val) {
                    temp.next = new ListNode(list1.val);
                    temp = temp.next;
                    list1 = list1.next;
                } else {
                    temp.next = new ListNode(list1.val);
                    temp.next.next = new ListNode(list1.val);
                    temp = temp.next.next;
                    list2 = list2.next;
                    list1 = list1.next;
                }
            } else if (list1 == null) {
                temp.next = list2;
                list2 = null;
            } else {
                temp.next = list1;
                list1 = null;
            }
        }
        return ans.next;
    }

    //两数相加
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        //返回的链表定义一个新联表伪指针，用来指向头指针，返回结果
        ListNode re = new ListNode(0);
        //临时下一个链表
        ListNode tem = re;
        //进位
        int carry = 0;
        //临时和temp以及当前节点值n1，n2
        int temp, n1, n2;
        while (l1 != null || l2 != null) {
            n1 = l1 != null ? l1.val : 0;
            n2 = l2 != null ? l2.val : 0;
            temp = n1 + n2 + carry;
            carry = temp / 10;
            temp = temp % 10;
            // 将求和数赋值给新链表的节点，
            // 注意这个时候不能直接将sum赋值给cur.next = sum。
            // 所以这个时候要创一个新的节点，将值赋予节点
            tem.next = new ListNode(temp);
            tem = tem.next;
            //当链表l1不等于null的时候，将l1 的节点后移
            if (l1 != null) {
                l1 = l1.next;
            }
            //当链表l2 不等于null的时候，将l2的节点后移
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry == 1) {
            tem.next = new ListNode(carry);
        }
        return re.next;
        //不能直接加，因为要补0，每一位加的时候算新节点
        /*int num=0;int index=0;
        while (l1!=null){
            num= (int) (num+l1.val*Math.pow(10,index));
            index++;
            l1=l1.next;
        }
        index = 0;
        while (l2!=null){
            num= (int) (num+l2.val*Math.pow(10,index));
            index++;
            l2=l2.next;
        }

        ListNode ans = new ListNode(0);
        //记住初始节点
        ListNode tem=ans;
        if(num==0)return ans;
        while (num!=0){
            tem.next=new ListNode(num%10);
            tem=tem.next;
            num/=10;
        }
        return ans.next;*/
    }

    //环形链表
    public static boolean hasCycle(ListNode head) {
        //快慢指针
        /*if(head==null||head.next==null)return false;
        ListNode slow = head;
        ListNode fast = head.next;
        while (fast!=slow){
            if(fast==null||fast.next==null)return false;
            slow=slow.next;
            fast=fast.next.next;
        }
        return true;*/
        //hash
        HashSet<ListNode> set = new HashSet<>();
        while (head != null) {
            if (!set.add(head)) return true;
            head = head.next;
        }
        return false;

    }

    public static int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        //或者linkedList

        for (String token : tokens) {
            if (token.equals("+")) {
                int ans = 0;
                ans = stack.pop() + stack.pop();
                stack.push(ans);
            } else if (token.equals("-")) {
                int h = stack.pop();
                h = stack.pop() - h;
                stack.push(h);
            } else if (token.equals("*")) {
                int x = stack.pop() * stack.pop();
                stack.push(x);
            } else if (token.equals("/")) {
                int c = stack.pop();
                c = stack.pop() / c;
                stack.push(c);
            } else stack.push(Integer.parseInt(token));
        }
        return stack.pop();
    }
}
