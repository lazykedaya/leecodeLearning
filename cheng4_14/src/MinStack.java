import org.w3c.dom.Node;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Stack;

public class MinStack {
    Deque<Integer>xStack;
    Deque<Integer>minStack;

    public MinStack() {
     xStack=new LinkedList<>();
     minStack=new LinkedList<>();
     minStack.push(Integer.MAX_VALUE);
    }

    public void push(int val) {
       xStack.push(val);
       minStack.push(Math.min(minStack.peek(),val));
    }

    public void pop() {
        xStack.pop();
        minStack.pop();
    }

    public int top() {
        return xStack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
