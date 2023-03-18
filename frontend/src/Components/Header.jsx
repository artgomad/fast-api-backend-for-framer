import React from "react";

const Header = () => {
  return (
    <nav style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.5rem", backgroundColor: "gray" }}>
      <div style={{ display: "flex", alignItems: "center" }}>
        <h1 style={{ fontSize: "1rem", marginRight: "0.5rem" }}>Todos</h1>
        <hr style={{ height: "1px", width: "100%", backgroundColor: "black", border: "none", margin: 0 }} />
      </div>
    </nav>
  );
};

export default Header;
