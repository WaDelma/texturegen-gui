use daggy::{Dag, Walker, NodeIndex, EdgeIndex, WouldCycle};
use daggy::petgraph::graph::IndexType;

struct Edge {
    source: u32,
    target: u32,
}

pub struct PortNumbered<N, Ix: IndexType = u32> {
    dag: Dag<N, Edge, Ix>,
}

impl<N, Ix: IndexType> PortNumbered<N, Ix> {
    pub fn new() -> PortNumbered<N, Ix> {
        PortNumbered {
            dag: Dag::new(),
        }
    }

    pub fn edges<'a>(&'a self) -> Edges<'a, Ix> {
        Edges(self.dag.raw_edges(), 0)
    }

    pub fn parents(&self, node: NodeIndex<Ix>) -> Parents<N, Ix> {
        Parents(&self.dag, self.dag.parents(node))
    }

    pub fn children(&self, node: NodeIndex<Ix>) -> Children<N, Ix> {
        Children(&self.dag, self.dag.children(node))
    }

    pub fn update_edge(&mut self, src: NodeIndex<Ix>, src_port: u32, trg: NodeIndex<Ix>, trg_port: u32) -> Result<EdgeIndex<Ix>, WouldBreak> {
        let replaced = self.dag.parents(trg).find_edge(&self.dag, |dag, e, _| dag.edge_weight(e).unwrap().target == trg_port);
        let result = self.dag.update_edge(src, trg, Edge{source: src_port, target: trg_port}).map_err(Into::into);
        if let Ok(_) = result {
            if let Some(e) = replaced {
                self.dag.remove_edge(e);
            }
        }
        result
    }

    pub fn remove_edge_to_port(&mut self, node: NodeIndex<Ix>, port: u32) -> Option<(NodeIndex<Ix>, u32)> {
        if let Some(e) = self.dag.parents(node).find_edge(&self.dag, |dag, e, _| dag.edge_weight(e).unwrap().target == port) {
            let result = (self.dag.edge_endpoints(e).unwrap().0, self.dag.edge_weight(e).unwrap().source);
            self.dag.remove_edge(e);
            Some(result)
        } else {
            None
        }
    }

    pub fn remove_outgoing_edges(&mut self, node: NodeIndex<Ix>) {
        let mut walker = self.dag.children(node);
        while let Some(e) = walker.next_edge(&self.dag) {
            self.dag.remove_edge(e);
        }
    }
}

impl<N, Ix: IndexType> PortNumbered<N, Ix> {
    pub fn add_node(&mut self, weight: N) -> NodeIndex<Ix> {
        self.dag.add_node(weight)
    }

    pub fn remove_node(&mut self, node: NodeIndex<Ix>) -> Option<N> {
        self.dag.remove_node(node)
    }

    pub fn node_weight(&self, node: NodeIndex<Ix>) -> Option<&N> {
        self.dag.node_weight(node)
    }

    pub fn node_weight_mut(&mut self, node: NodeIndex<Ix>) -> Option<&mut N> {
        self.dag.node_weight_mut(node)
    }

    pub fn raw_nodes(&self) -> ::daggy::RawNodes<N, Ix> {
        self.dag.raw_nodes()
    }

    pub fn edge_count(&self) -> usize {
        self.dag.edge_count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WouldBreak {
    WouldCycle,
    WouldUnport,
}

impl From<WouldCycle<Edge>> for WouldBreak {
    fn from(_: WouldCycle<Edge>) -> Self {
        WouldBreak::WouldCycle
    }
}

pub struct Edges<'a, Ix: IndexType>(::daggy::RawEdges<'a, Edge, Ix>, usize);

impl<'a, Ix: IndexType> Iterator for Edges<'a, Ix> {
    type Item = (NodeIndex<Ix>, u32, NodeIndex<Ix>, u32);
    fn next(&mut self) -> Option<Self::Item> {
        if self.1 < self.0.len() {
            let e = &self.0[self.1];
            self.1 += 1;
            Some((e.source(), e.weight.source, e.target(), e.weight.target))
        } else {
            None
        }
    }
}

pub struct Parents<'a, N: 'a, Ix: IndexType>(&'a Dag<N, Edge, Ix>, ::daggy::Parents<N, Edge, Ix>);

impl<'a, N: 'a, Ix: IndexType> Iterator for Parents<'a, N, Ix> {
    type Item = (NodeIndex<Ix>, u32, u32);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((edge, node)) = self.1.next(self.0) {
            let edge = self.0.edge_weight(edge).unwrap();
            Some((node, edge.source, edge.target))
        } else {
            None
        }
    }
}

pub struct Children<'a, N: 'a, Ix: IndexType>(&'a Dag<N, Edge, Ix>, ::daggy::Children<N, Edge, Ix>);

impl<'a, N: 'a, Ix: IndexType> Iterator for Children<'a, N, Ix> {
    type Item = (u32, NodeIndex<Ix>, u32);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((edge, node)) = self.1.next(self.0) {
            let edge = self.0.edge_weight(edge).unwrap();
            Some((edge.source, node, edge.target))
        } else {
            None
        }
    }
}
