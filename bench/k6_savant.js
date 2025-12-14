import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "20s", target: 5 },
    { duration: "40s", target: 25 },
    { duration: "60s", target: 50 },
    { duration: "40s", target: 25 },
    { duration: "20s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate<0.005"],
    http_req_duration: ["p(95)<600", "p(99)<900"],
  },
};

export default function () {
  const base = __ENV.BASE_URL;
  const payload = JSON.stringify({
    prompt: "Explain Savant RRF briefly.",
    answer: "Savant evaluates semantic quality with RRF meta-logic."
  });

  const res = http.post(`${base}/judge`, payload, {
    headers: { "Content-Type": "application/json" },
    timeout: "60s",
  });

  check(res, {
    "status 200": (r) => r.status === 200,
  });

  sleep(0.2);
}
