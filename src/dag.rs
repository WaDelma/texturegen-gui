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

    pub fn add_edge(&mut self, src: NodeIndex<Ix>, src_port: u32, trg: NodeIndex<Ix>, trg_port: u32) -> Result<EdgeIndex<Ix>, WouldBreak> {
        if let None = self.dag.parents(trg).find_edge(&self.dag, |dag, e, _| dag.edge_weight(e).unwrap().target == trg_port) {
            self.dag.update_edge(src, trg, Edge{source: src_port, target: trg_port}).map_err(Into::into)
        } else {
            Err(WouldBreak::WouldUnport)
        }
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
}

impl<N, Ix: IndexType> PortNumbered<N, Ix> {
    pub fn add_node(&mut self, weight: N) -> NodeIndex<Ix> {
        self.dag.add_node(weight)
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
