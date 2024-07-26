public class ListNode {
    //单向链表，每一个结点只知道自己和下一个结点的数
    int val;
    ListNode next;
    ListNode(int val,ListNode next){
        this.val=val;
        this.next=next;
    }

    public ListNode(int val) {
        this.val = val;
    }
    public ListNode() {
    }
}
