import { useCallback, useEffect, useState } from "react";
import { UserPlus, Users } from "lucide-react";

import { errorMessage } from "../api/client";
import { addMember, createTeam, listTeams, removeMember, teamUsage } from "../api/teamsApi";
import { useAuth } from "../context/AuthContext";
import { formatCount } from "../utils/format";

function TeamCard({ team, me, onChange }) {
  const [email, setEmail] = useState("");
  const [usage, setUsage] = useState(null);
  const [error, setError] = useState("");
  const owner = team.members.some((m) => m.user_id === me.id && m.role === "owner");

  useEffect(() => {
    teamUsage(team.id).then(setUsage).catch(() => {});
  }, [team.id]);

  const act = async (action) => {
    setError("");
    try {
      await action();
      onChange();
    } catch (caught) {
      setError(errorMessage(caught));
    }
  };

  return (
    <section className="panel section">
      <h2>{team.name}</h2>
      {usage && (
        <p className="muted small">
          {formatCount(Math.round(usage.total_units))} compute units used in the team&apos;s sandboxes
        </p>
      )}
      <table className="table compact">
        <tbody>
          {team.members.map((member) => (
            <tr key={member.user_id}>
              <td>{member.email}</td>
              <td>{member.role}</td>
              <td>
                {(owner || member.user_id === me.id) && (
                  <button
                    type="button"
                    className="link-button"
                    onClick={() => act(() => removeMember(team.id, member.user_id))}
                  >
                    {member.user_id === me.id ? "Leave" : "Remove"}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {owner && (
        <form
          className="inline-form compact"
          onSubmit={(event) => {
            event.preventDefault();
            act(async () => {
              await addMember(team.id, email);
              setEmail("");
            });
          }}
        >
          <input type="email" required placeholder="colleague@example.com" value={email} onChange={(e) => setEmail(e.target.value)} />
          <button className="button" type="submit">
            <UserPlus size={16} aria-hidden="true" /> Add member
          </button>
        </form>
      )}
      {error && <p className="error small">{error}</p>}
    </section>
  );
}

export default function TeamsPage() {
  const { user } = useAuth();
  const [teams, setTeams] = useState(null);
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(() => listTeams().then(setTeams).catch((caught) => setError(errorMessage(caught))), []);
  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Teams</h1>
          <p className="muted">
            Members use the team&apos;s sandboxes and datasets; only the person who created a sandbox or dataset can
            delete it.
          </p>
        </div>
      </header>
      <form
        className="panel inline-form"
        onSubmit={async (event) => {
          event.preventDefault();
          try {
            await createTeam(name);
            setName("");
            load();
          } catch (caught) {
            setError(errorMessage(caught));
          }
        }}
      >
        <label className="field grow">
          <span>New team</span>
          <input required value={name} placeholder="e.g. imaging lab" onChange={(e) => setName(e.target.value)} />
        </label>
        <button className="button primary" type="submit">
          <Users size={16} aria-hidden="true" /> Create
        </button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {teams?.length === 0 && <p className="muted">No teams yet.</p>}
      {teams?.map((team) => (
        <TeamCard key={team.id} team={team} me={user} onChange={load} />
      ))}
    </div>
  );
}
