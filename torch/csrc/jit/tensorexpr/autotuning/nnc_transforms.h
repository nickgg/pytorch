#pragma once

#include <torch/csrc/jit/tensorexpr/autotuning/autotune_base.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace tuning {
namespace transforms {

struct SplitWithMask : public Transform {
  std::string loopVar;
  int factor;
  SplitWithMask(std::string l, int f) : loopVar(l), factor(f) {}

  void log() const override {
    std::cout << "SplitWithMask(" << loopVar << ", " << factor << ")";
  }

  std::string name() const override {
    return "SplitWithMask";
  }

  Transform* clone() const override {
    return new SplitWithMask(loopVar, factor);
  }
};

struct SplitWithTail : public Transform {
  std::string loopVar;
  int factor;
  SplitWithTail(std::string l, int f) : loopVar(l), factor(f) {}

  void log() const override {
    std::cout << "SplitWithTail(" << loopVar << ", " << factor << ")";
  }

  std::string name() const override {
    return "SplitWithTail";
  }

  Transform* clone() const override {
    return new SplitWithTail(loopVar, factor);
  }
};

struct ReorderAxis : public Transform {
  std::string loop1;
  std::string loop2;
  ReorderAxis(std::string l1, std::string l2) : loop1(l1), loop2(l2) {}

  void log() const override {
    std::cout << "ReorderAxis(" << loop1 << ", " << loop2 << ")";
  }

  std::string name() const override {
    return "ReorderAxis";
  }

  Transform* clone() const override {
    return new ReorderAxis(loop1, loop2);
  }
};

struct Inline : public Transform {
  std::string buf;
  Inline(std::string b) : buf(b) {}

  void log() const override {
    std::cout << "Inline(" << buf << ")";
  }

  std::string name() const override {
    return "Inline";
  }

  Transform* clone() const override {
    return new Inline(buf);
  }
};

struct Rfactor : public Transform {
  std::string buf;
  std::string var;
  Rfactor(std::string b, std::string v) : buf(b), var(v) {}

  void log() const override {
    std::cout << "Rfactor(" << buf << ", " << var << ")";
  }

  std::string name() const override {
    return "Rfactor";
  }

  Transform* clone() const override {
    return new Rfactor(buf, var);
  }
};

struct ComputeAt : public Transform {
  int stmt_idx;
  std::string loopVar;
  ComputeAt(int s, std::string l) : stmt_idx(s), loopVar(l) {}

  void log() const override {
    std::cout << "ComputeAt(" << stmt_idx << ", " << loopVar << ")";
  }

  std::string name() const override {
    return "ComputeAt";
  }

  Transform* clone() const override {
    return new ComputeAt(stmt_idx, loopVar);
  }
};

struct Vectorize : public Transform {
  std::string loopVar;
  Vectorize(std::string l) : loopVar(l) {}

  void log() const override {
    std::cout << "Vectorize(" << loopVar << ")";
  }

  std::string name() const override {
    return "Vectorize";
  }

  Transform* clone() const override {
    return new Vectorize(loopVar);
  }
};

struct BindBlockIdx : public Transform {
  std::string loopVar;
  int idx;
  BindBlockIdx(std::string l, int i) : loopVar(l), idx(i) {}

  void log() const override {
    std::cout << "BindBlockIdx(" << loopVar << ", " << idx << ")";
  }

  std::string name() const override {
    return "BindBlockIdx";
  }

  Transform* clone() const override {
    return new BindBlockIdx(loopVar, idx);
  }
};

struct BindThreadIdx : public Transform {
  std::string loopVar;
  int idx;
  BindThreadIdx(std::string l, int i) : loopVar(l), idx(i) {}

  void log() const override {
    std::cout << "BindThreadIdx(" << loopVar << ", " << idx << ")";
  }

  std::string name() const override {
    return "BindThreadIdx";
  }

  Transform* clone() const override {
    return new BindThreadIdx(loopVar, idx);
  }
};

struct SwapAxisIdx : public Transform {
  std::string before;
  std::string after;
  SwapAxisIdx(std::string b, std::string a) : before(b), after(a) {}

  void log() const override {
    std::cout << "SwapAxisIdx(" << before << ", " << after << ")";
  }

  std::string name() const override {
    return "SwapAxisIdx";
  }

  Transform* clone() const override {
    return new SwapAxisIdx(before, after);
  }
};

} // namespace transforms
} // namespace tuning
} // namespace tensorexpr
} // namespace jit
} // namespace torch

