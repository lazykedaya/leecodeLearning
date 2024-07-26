
/**
 * 链表题*/
public class test1 {
    public static void main(String[] args) {
        ListNode l1 = new ListNode(2, new ListNode(3, null));
//        ListNode l1 = new ListNode(5, l0);
        ListNode l2 = new ListNode(2, new ListNode(5, null));


        //返回的链表定义一个新联表伪指针，用来指向头指针，返回结果
        ListNode re=new ListNode(0);
        //临时下一个链表
        ListNode tem=re;
        //进位
        int carry = 0;
        //临时和temp以及当前节点值n1，n2
        int temp,n1,n2;
        while (l1 != null || l2 != null) {
            n1=l1!=null?l1.val:0;
            n2=l2!=null?l2.val:0;
            temp=n1+n2+carry;
            carry=temp/10;
            temp=temp%10;
            // 将求和数赋值给新链表的节点，
            // 注意这个时候不能直接将sum赋值给cur.next = sum。
            // 所以这个时候要创一个新的节点，将值赋予节点
            tem.next=new ListNode(temp);
            tem=tem.next;
            //当链表l1不等于null的时候，将l1 的节点后移
            if(l1 !=null){
                l1 = l1.next;
            }
            //当链表l2 不等于null的时候，将l2的节点后移
            if(l2 !=null){
                l2 = l2.next;
            }
        }
        if(carry == 1){
            tem.next = new ListNode(carry);
        }
        //返回链表的头节点
        while (re.next!=null){
            System.out.println(re.next.val);
            re=re.next;
        }



    }
}
