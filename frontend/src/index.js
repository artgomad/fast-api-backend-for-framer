import React from "react";
import { render } from 'react-dom';

import Header from "./components/Header";
import Todos from "./components/Todos";

function App() {
  return (
    <>
      <Header />
      <Todos />
     
    </>
  )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)
