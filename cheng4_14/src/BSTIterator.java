import java.util.ArrayList;
import java.util.LinkedList;

public class BSTIterator {
    static ArrayList<Integer>list;
    static int k;
    public BSTIterator(TreeNode root) {
        LinkedList<TreeNode>stack = new LinkedList<>();
        list=new ArrayList<>();
        k=0;
        list.add(-1);
        while (root != null || !stack.isEmpty()) {
            if (root != null) {
                stack.push(root);
                root = root.left;
            } else { //pNode == null && !stack.isEmpty()
                TreeNode node1 = stack.pop();
                list.add(node1.val);
                root = node1.right;
            }
        }
    }

    public int next() {
        return list.get(++k);
    }

    public boolean hasNext() {
        int size = list.size();
        if(k+1<size)return true;
        else return false;
    }
}
